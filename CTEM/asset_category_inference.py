"""
Asset Category Inference — MLP Classifier
==========================================

Predicts asset categories (multi-label) from structured asset features
using the trained MLP model (best params from RandomizedSearchCV).

Model: MLPClassifier(512,256,128), alpha=0.0005, lr=0.001, batch=512, relu
F1 Micro on test set: 0.7919

Input features (148 total):
  - Ports (open port numbers)
  - Banners (raw HTTP banner string)
  - Domain name
  - Technologies (list of tech names)
  - Passive DNS query count

Usage:
    # Single asset
    python asset_category_inference.py

    # As a module
    from asset_category_inference import AssetCategoryClassifier
    clf = AssetCategoryClassifier()
    results = clf.predict({
        'ports': [80, 443],
        'banners': 'HTTP/1.1 200 OK\\r\\nServer: nginx\\r\\nContent-Type: text/html',
        'domain_name': 'api.example.com',
        'technologies': ['nginx'],
        'passive_dns_query_count': 150,
    })
"""

import os
import re
import sys
import json
import pickle
import warnings

# ── numpy 2.x → 1.x compatibility shim ──
# Artifacts saved with numpy 2.x reference `numpy._core` internally;
# numpy 1.x (e.g. Databricks runtime) only has `numpy.core`.
# Custom Unpickler remaps module names during deserialization.
import numpy as np


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that remaps numpy._core → numpy.core for numpy 1.x compat."""
    def find_class(self, module: str, name: str):
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def _load_pkl(path: str):
    """Load a pickle file with numpy 2→1 compatibility."""
    with open(path, 'rb') as f:
        if hasattr(np, '_core'):
            return pickle.load(f)
        return _NumpyCompatUnpickler(f).load()


from scipy.sparse import hstack

warnings.filterwarnings('ignore')

# ── Path to saved model artifacts ──
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_artifacts')


class AssetCategoryClassifier:
    """Multi-label asset category classifier using a trained MLP model."""

    def __init__(self, artifacts_dir: str = ARTIFACTS_DIR):
        """Load all model artifacts from disk."""
        self.model = _load_pkl(os.path.join(artifacts_dir, 'mlp_model.pkl'))
        self.scaler = _load_pkl(os.path.join(artifacts_dir, 'scaler.pkl'))
        self.mlb = _load_pkl(os.path.join(artifacts_dir, 'mlb.pkl'))
        self.tfidf_word = _load_pkl(os.path.join(artifacts_dir, 'tfidf_word.pkl'))
        self.tfidf_char = _load_pkl(os.path.join(artifacts_dir, 'tfidf_char.pkl'))
        self.svd = _load_pkl(os.path.join(artifacts_dir, 'svd.pkl'))
        self.top_ports = _load_pkl(os.path.join(artifacts_dir, 'top_ports.pkl'))
        self.top_servers = _load_pkl(os.path.join(artifacts_dir, 'top_servers.pkl'))
        self.all_techs = _load_pkl(os.path.join(artifacts_dir, 'all_techs.pkl'))
        self.feature_columns = _load_pkl(os.path.join(artifacts_dir, 'feature_columns.pkl'))

        # Categories the model can predict
        self.categories = list(self.mlb.classes_)

    # ──────────────────────────────────────────────
    # Banner parsing
    # ──────────────────────────────────────────────
    @staticmethod
    def _extract_banner_features(banner_raw: str) -> dict:
        """Extract Server header, HTTP status code, and Content-Type from a raw banner."""
        if not banner_raw or str(banner_raw).strip() in ('', '{}'):
            return {'server': '', 'status_code': '', 'content_type': ''}

        text = str(banner_raw)

        m = re.search(r'Server:\s*([^\r\n"\\]+)', text, re.IGNORECASE)
        server = m.group(1).strip().lower() if m else ''

        m = re.search(r'HTTP/[\d.]+\s+(\d{3})', text)
        status = m.group(1) if m else ''

        m = re.search(r'Content-Type:\s*([^\r\n";\\]+)', text, re.IGNORECASE)
        ctype = m.group(1).strip().lower() if m else ''

        return {'server': server, 'status_code': status, 'content_type': ctype}

    # ──────────────────────────────────────────────
    # Feature engineering for a single asset
    # ──────────────────────────────────────────────
    def _build_features(self, asset: dict) -> np.ndarray:
        """
        Build the 148-dim feature vector for a single asset.

        Parameters
        ----------
        asset : dict with keys:
            - ports : list[int|str]       — open port numbers
            - banners : str               — raw HTTP banner string
            - domain_name : str           — e.g. 'api.example.com'
            - technologies : list[str]    — e.g. ['nginx', 'CloudFlare']
            - passive_dns_query_count : int — DNS query count
        """
        # ── 1. Port features (31) ──
        ports = [str(p) for p in (asset.get('ports') or [])]
        port_feats = {}
        for p in self.top_ports:
            port_feats[f'port_{p}'] = 1 if p in ports else 0
        port_feats['port_count'] = len(ports)

        # ── 2. Banner features (28) ──
        banner = self._extract_banner_features(asset.get('banners', ''))
        banner_feats = {}
        for s in self.top_servers:
            col = f'srv_{s[:30]}'
            banner_feats[col] = 1 if banner['server'] == s else 0

        status = banner['status_code']
        for grp in ['2xx', '3xx', '4xx', '5xx']:
            banner_feats[f'status_{grp}'] = 1 if (status and status[0] + 'xx' == grp) else 0

        ctype = banner['content_type']
        for ct in ['text/html', 'text/plain', 'application/json', 'application/xml']:
            col = f'ctype_{ct.replace("/", "_")}'
            banner_feats[col] = 1 if ctype == ct else 0

        # ── 3. Domain SVD features (50) ──
        domain = asset.get('domain_name', '') or ''
        domain_tokens = ' '.join(re.split(r'[.\-_]', str(domain).lower()))

        X_word = self.tfidf_word.transform([domain_tokens])
        X_char = self.tfidf_char.transform([domain_tokens])
        X_domain_tfidf = hstack([X_word, X_char])
        X_domain_svd = self.svd.transform(X_domain_tfidf)

        domain_feats = {}
        for i in range(50):
            domain_feats[f'dom_svd_{i}'] = X_domain_svd[0, i]

        # ── 4. Technology features (37) ──
        techs = asset.get('technologies') or []
        tech_feats = {}
        for t in self.all_techs:
            col = f'tech_{t[:25]}'
            tech_feats[col] = 1 if t in techs else 0

        # ── 5. DNS features (2) ──
        dns_count = int(asset.get('passive_dns_query_count', 0) or 0)
        dns_feats = {
            'dns_log': np.log1p(dns_count),
            'dns_has': 1 if dns_count > 0 else 0,
        }

        # ── Combine in correct column order ──
        all_feats = {}
        all_feats.update(port_feats)
        all_feats.update(banner_feats)
        all_feats.update(domain_feats)
        all_feats.update(tech_feats)
        all_feats.update(dns_feats)

        feature_vector = np.array(
            [all_feats.get(col, 0.0) for col in self.feature_columns],
            dtype=np.float32,
        ).reshape(1, -1)

        return feature_vector

    # ──────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────
    def predict(self, asset: dict) -> dict:
        """
        Predict categories for a single asset.

        Parameters
        ----------
        asset : dict — see _build_features() for expected keys.

        Returns
        -------
        dict with:
            - categories : list[str]  — predicted category labels
            - probabilities : dict[str, float] — probability for each category
        """
        X = self._build_features(asset)
        X_scaled = self.scaler.transform(X)

        # Binary predictions
        y_pred = self.model.predict(X_scaled)[0]

        # Probability estimates
        probas = self.model.predict_proba(X_scaled)
        # MLP multi-label: predict_proba returns list of arrays, one per output
        if isinstance(probas, list):
            # Each element is (1, 2) array for binary classification per label
            proba_values = [p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in probas]
        else:
            proba_values = probas[0].tolist()

        predicted_labels = [
            self.categories[i] for i, val in enumerate(y_pred) if val == 1
        ]

        proba_dict = {
            self.categories[i]: round(float(proba_values[i]), 4)
            for i in range(len(self.categories))
        }

        # Sort probabilities descending
        proba_sorted = dict(sorted(proba_dict.items(), key=lambda x: -x[1]))

        return {
            'categories': predicted_labels,
            'probabilities': proba_sorted,
        }

    def predict_batch(self, assets: list[dict]) -> list[dict]:
        """
        Predict categories for a batch of assets.

        Parameters
        ----------
        assets : list[dict] — list of asset dicts.

        Returns
        -------
        list[dict] — one result dict per asset.
        """
        if not assets:
            return []

        # Build feature matrix for all assets at once
        X = np.vstack([self._build_features(a) for a in assets])
        X_scaled = self.scaler.transform(X)

        y_preds = self.model.predict(X_scaled)
        probas_all = self.model.predict_proba(X_scaled)

        results = []
        for idx in range(len(assets)):
            y_pred = y_preds[idx]

            if isinstance(probas_all, list):
                proba_values = [
                    p[idx, 1] if p.shape[1] > 1 else p[idx, 0]
                    for p in probas_all
                ]
            else:
                proba_values = probas_all[idx].tolist()

            predicted_labels = [
                self.categories[i] for i, val in enumerate(y_pred) if val == 1
            ]

            proba_dict = {
                self.categories[i]: round(float(proba_values[i]), 4)
                for i in range(len(self.categories))
            }
            proba_sorted = dict(sorted(proba_dict.items(), key=lambda x: -x[1]))

            results.append({
                'categories': predicted_labels,
                'probabilities': proba_sorted,
            })

        return results


# ──────────────────────────────────────────────────────────
# CLI interface
# ──────────────────────────────────────────────────────────
def main():
    """Interactive CLI for single-asset classification."""
    print("=" * 60)
    print("  Asset Category Classifier — MLP Inference")
    print("  Model: MLP (512,256,128) | 44 categories | 148 features")
    print("=" * 60)

    clf = AssetCategoryClassifier()
    print(f"✅ Model loaded ({len(clf.categories)} categories)\n")

    while True:
        print("-" * 60)
        domain = input("Domain name (or 'quit'): ").strip()
        if domain.lower() in ('quit', 'exit', 'q'):
            break

        ports_raw = input("Ports (comma-separated, e.g. 80,443): ").strip()
        ports = [p.strip() for p in ports_raw.split(',') if p.strip()] if ports_raw else []

        banner = input("Banner (raw HTTP response, or empty): ").strip()

        techs_raw = input("Technologies (comma-separated, or empty): ").strip()
        techs = [t.strip() for t in techs_raw.split(',') if t.strip()] if techs_raw else []

        dns_raw = input("Passive DNS query count (integer, or 0): ").strip()
        dns_count = int(dns_raw) if dns_raw.isdigit() else 0

        asset = {
            'ports': ports,
            'banners': banner,
            'domain_name': domain,
            'technologies': techs,
            'passive_dns_query_count': dns_count,
        }

        result = clf.predict(asset)

        print(f"\n🏷  Predicted categories: {result['categories']}")
        print(f"\n📊 Top 10 probabilities:")
        for i, (cat, prob) in enumerate(result['probabilities'].items()):
            if i >= 10:
                break
            bar = '█' * int(prob * 40)
            print(f"  {cat:<35} {prob:.4f}  {bar}")
        print()


if __name__ == '__main__':
    main()
