"""Quick smoke-test for asset_category_inference.py"""
from asset_category_inference import AssetCategoryClassifier

clf = AssetCategoryClassifier()
print(f"Model loaded – {len(clf.categories)} categories")

# --- Test 1: nginx web server ------------------------------------------
r = clf.predict({
    "ports": [80, 443],
    "banners": "HTTP/1.1 200 OK\r\nServer: nginx\r\nContent-Type: text/html",
    "domain_name": "www.example.com",
    "technologies": ["nginx"],
    "passive_dns_query_count": 500,
})
print(f"\nTest 1 (nginx web server): {r['categories']}")
for c, p in list(r["probabilities"].items())[:5]:
    print(f"  {c:<35} {p:.4f}")

# --- Test 2: Akamai CDN ------------------------------------------------
r = clf.predict({
    "ports": [443, 80],
    "banners": "HTTP/1.1 403 Forbidden\r\nServer: AkamaiGHost",
    "domain_name": "cdn.assets.company.com",
    "technologies": ["AkamaiGHost"],
    "passive_dns_query_count": 1200,
})
print(f"\nTest 2 (Akamai CDN): {r['categories']}")
for c, p in list(r["probabilities"].items())[:5]:
    print(f"  {c:<35} {p:.4f}")

# --- Test 3: Mail server -----------------------------------------------
r = clf.predict({
    "ports": [25, 587, 993],
    "banners": "",
    "domain_name": "mail.company.org",
    "technologies": ["Postfix smtpd"],
    "passive_dns_query_count": 80,
})
print(f"\nTest 3 (mail server): {r['categories']}")
for c, p in list(r["probabilities"].items())[:5]:
    print(f"  {c:<35} {p:.4f}")

# --- Test 4: batch prediction -------------------------------------------
results = clf.predict_batch([
    {
        "ports": [443],
        "banners": "",
        "domain_name": "api.stripe.com",
        "technologies": [],
        "passive_dns_query_count": 200,
    },
    {
        "ports": [22],
        "banners": "",
        "domain_name": "bastion.internal.net",
        "technologies": ["OpenSSH"],
        "passive_dns_query_count": 5,
    },
])
print("\nTest 4 (batch):")
for i, r in enumerate(results):
    print(f"  Asset {i+1}: {r['categories']}")
