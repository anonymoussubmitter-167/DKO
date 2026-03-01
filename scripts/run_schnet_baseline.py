#!/usr/bin/env python3
"""
SchNet baseline for 3D molecular property prediction.
Addresses Critique #3: SOTA 3D GNN baselines.
"""

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from rdkit import Chem
from rdkit.Chem import AllChem

DATASETS = ["esol", "freesolv", "lipophilicity"]
SEEDS = [42, 123, 456]


def smiles_to_pyg_data(smiles: str, target: float, max_atoms: int = 100) -> Data:
    """Convert SMILES to PyG Data object with 3D coordinates."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    try:
        # Use ETKDG for better conformer quality
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # Fallback to basic embedding
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass  # OK if optimization fails, coordinates are still valid
    except Exception:
        return None

    if mol.GetNumConformers() == 0:
        return None

    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()

    if n_atoms > max_atoms:
        return None

    # Atomic numbers
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)

    # 3D positions
    pos = torch.tensor([list(conf.GetAtomPosition(i)) for i in range(n_atoms)], dtype=torch.float)

    # Target
    y = torch.tensor([target], dtype=torch.float)

    return Data(z=z, pos=pos, y=y)


def load_dataset_pyg(dataset_name: str, split: str = "train"):
    """Load dataset and convert to PyG format."""
    data_path = Path(f"data/conformers/{dataset_name}/{split}.pkl")
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    smiles_list = data["smiles"]
    targets = np.array(data["labels"]).squeeze()

    pyg_data = []
    for smi, target in zip(smiles_list, targets):
        d = smiles_to_pyg_data(smi, target)
        if d is not None:
            pyg_data.append(d)

    return pyg_data


def radius_graph_pure(pos, batch, cutoff=10.0, max_num_neighbors=32):
    """Pure PyTorch radius graph computation (no torch-cluster needed)."""
    device = pos.device
    # Compute pairwise distances
    n = pos.size(0)
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    dist = torch.cdist(pos.unsqueeze(0), pos.unsqueeze(0)).squeeze(0)

    # Mask: same batch, within cutoff, not self-loops
    same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)
    within_cutoff = dist < cutoff
    not_self = ~torch.eye(n, dtype=torch.bool, device=device)
    mask = same_batch & within_cutoff & not_self

    # Limit neighbors per node
    if max_num_neighbors < n:
        # Set masked-out distances to inf for top-k
        dist_masked = dist.clone()
        dist_masked[~mask] = float('inf')
        # For each node, keep only max_num_neighbors closest
        _, topk_idx = dist_masked.topk(min(max_num_neighbors, n - 1), dim=1, largest=False)
        topk_mask = torch.zeros_like(mask)
        topk_mask.scatter_(1, topk_idx, True)
        mask = mask & topk_mask

    edge_index = mask.nonzero(as_tuple=False).t().contiguous()
    # edge_index[0] = target, edge_index[1] = source (PyG convention: col -> row)
    return edge_index.flip(0)


class SchNetWrapper(nn.Module):
    """SchNet-like model using pure PyTorch (no torch-cluster dependency).

    Implements continuous-filter convolution with RBF expansion and
    interaction blocks following the SchNet architecture.
    """

    def __init__(self, hidden_channels=128, num_filters=128, num_interactions=6,
                 num_gaussians=50, cutoff=10.0, max_z=100):
        super().__init__()
        self.cutoff = cutoff
        self.hidden_channels = hidden_channels

        # Atom embedding
        self.embedding = nn.Embedding(max_z, hidden_channels)

        # Gaussian RBF expansion
        self.num_gaussians = num_gaussians
        offset = torch.linspace(0, cutoff, num_gaussians)
        self.register_buffer('offset', offset)
        self.width = (offset[1] - offset[0]).item()

        # Interaction blocks
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            self.interactions.append(SchNetInteraction(
                hidden_channels, num_filters, num_gaussians, cutoff
            ))

        # Output MLP
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, z, pos, batch):
        # Get edge index via pure-pytorch radius graph
        edge_index = radius_graph_pure(pos, batch, self.cutoff)

        if edge_index.size(1) == 0:
            # No edges: return zeros
            n_graphs = batch.max().item() + 1
            return torch.zeros(n_graphs, 1, device=pos.device)

        # Compute edge distances
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)

        # RBF expansion
        rbf = torch.exp(-0.5 * ((dist.unsqueeze(-1) - self.offset) / self.width) ** 2)

        # Cosine cutoff
        cutoff_val = 0.5 * (torch.cos(dist * 3.14159265 / self.cutoff) + 1.0)
        cutoff_val = cutoff_val * (dist < self.cutoff).float()

        # Atom embeddings
        h = self.embedding(z)

        # Interaction blocks
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, rbf, cutoff_val)

        # Per-atom output
        out = self.output(h).squeeze(-1)

        # Sum pooling per molecule
        n_graphs = batch.max().item() + 1
        result = torch.zeros(n_graphs, device=out.device)
        result.scatter_add_(0, batch, out)

        return result


class SchNetInteraction(nn.Module):
    """Single SchNet interaction block."""

    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters),
            nn.SiLU(),
            nn.Linear(num_filters, num_filters),
        )
        self.lin1 = nn.Linear(hidden_channels, num_filters)
        self.lin2 = nn.Linear(num_filters, hidden_channels)
        self.act = nn.SiLU()

    def forward(self, x, edge_index, rbf, cutoff_val):
        row, col = edge_index
        num_nodes = x.size(0)
        num_filters = self.lin1.out_features

        # Filter generation
        W = self.mlp(rbf) * cutoff_val.unsqueeze(-1)

        # Message passing
        x_j = self.lin1(x[col])
        msg = x_j * W

        # Aggregate into num_filters-dimensional space
        agg = torch.zeros(num_nodes, num_filters, device=x.device)
        agg.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)

        return self.act(self.lin2(agg))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.z, data.pos, data.batch)
        target = data.y.squeeze()
        if target.dim() == 0:
            target = target.unsqueeze(0)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.z, data.pos, data.batch)
            out_np = out.squeeze(-1).cpu().numpy()
            y_np = data.y.squeeze(-1).cpu().numpy()
            # Handle scalar case (single molecule batch)
            if out_np.ndim == 0:
                out_np = out_np.reshape(1)
            if y_np.ndim == 0:
                y_np = y_np.reshape(1)
            preds.extend(out_np.tolist())
            targets.extend(y_np.tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)),
    }


def run_experiment(dataset_name: str, seed: int, device: str, epochs: int = 200):
    """Run single SchNet experiment."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"  Loading {dataset_name}...")
    train_data = load_dataset_pyg(dataset_name, "train")
    val_data = load_dataset_pyg(dataset_name, "val")
    test_data = load_dataset_pyg(dataset_name, "test")

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError(f"Empty split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    model = SchNetWrapper().to(device)
    optimizer = Adam(model.parameters(), lr=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["rmse"])

        if val_metrics["rmse"] < best_val_loss:
            best_val_loss = val_metrics["rmse"]
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 20:
            print(f"  Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_rmse={val_metrics['rmse']:.4f}")

    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, test_loader, device)

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--output-dir", default="results/schnet_baseline")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SchNet Baseline Experiments")
    print(f"Device: {device}")
    print(f"Datasets: {args.datasets}")
    print(f"Seeds: {args.seeds}")
    print("=" * 60)

    results = []

    for dataset in args.datasets:
        print(f"\n=== {dataset.upper()} ===")
        for seed in args.seeds:
            print(f"\nSeed {seed}:")
            try:
                metrics = run_experiment(dataset, seed, device, args.epochs)
                results.append({
                    "dataset": dataset,
                    "seed": seed,
                    "model": "schnet",
                    **metrics
                })
                print(f"  Test: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}")

                # Save incrementally
                with open(output_dir / "schnet_results_partial.json", "w") as f:
                    json.dump({"results": results}, f, indent=2)

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset in args.datasets:
        ds_results = [r for r in results if r["dataset"] == dataset]
        if ds_results:
            rmses = [r["rmse"] for r in ds_results]
            print(f"{dataset}: RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")

    # Save results
    with open(output_dir / "schnet_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/schnet_results.json")


if __name__ == "__main__":
    main()
