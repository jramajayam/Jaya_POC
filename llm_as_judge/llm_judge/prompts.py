"""
Prompts for independent LLM classification of network assets.

The LLM receives ONLY the 5 asset fields and classifies independently.
Comparison with MLP predictions happens in Python (see judge.py).

All prompts are self-contained — no external imports needed.
"""


# ── Category Definitions ──────────────────────────────────────────────────────

CATEGORY_DEFINITIONS = """
CATEGORY REFERENCE (grouped by function):

── Web & Content ──
• web server           — serves HTTP/HTTPS content (HTML, APIs). Default category for ANY asset with HTTP exposure, including bots, crawlers, CDN nodes, internal infra, or specialized services. In CTEM, "web server" means "has an HTTP-facing attack surface".
• content delivery network — CDN edge nodes distributing cached content (Akamai, Cloudflare, Fastly, Apple CDN patterns like "phobos", "ls.apple.com")
• content management system — CMS platforms (WordPress, Drupal, Joomla, etc.)
• media streaming device — audio/video streaming services or infrastructure
• object storage        — cloud object stores serving files (S3, GCS, Azure Blob)

── API & Application ──
• api endpoint          — programmatic interfaces (REST, GraphQL, gRPC). Look for "api" in domain, JSON responses, or API-specific ports.
• application server    — backend application runtimes (Tomcat, JBoss, Node.js app servers)
• software update server — serves software patches/updates (domain keywords: "update", "su.", "swscan")

── Communication ──
• mail server           — SMTP/IMAP/POP3 mail services. Look for port 25/465/587/993, "mail" in domain, Exchange/Postfix banners.
• chat server           — real-time messaging (Slack, Teams, XMPP, custom chat)
• notification server   — push notification services (domain keywords: "push", "notify", "courier")
• voip server           — voice-over-IP telephony services
• message broker        — message queue systems (RabbitMQ, Kafka, Redis pub/sub)

── Infrastructure & Networking ──
• dns server            — DNS resolution services (port 53)
• dhcp server           — dynamic IP assignment
• load balancer         — distributes traffic across backends
• reverse proxy         — proxies requests to backend servers (nginx proxy, HAProxy)
• web proxy             — forward proxy for client traffic
• firewall              — network security appliance
• router                — network routing device
• vpn server            — VPN gateway services
• time server           — NTP time synchronization (port 123)

── Security & Identity ──
• authentication server — SSO, OAuth, LDAP, identity providers
• certificate validation server — OCSP responders, CRL distribution, "certs" or "ocsp" in domain

── DevOps & Development ──
• continuous integration server — CI/CD systems (Jenkins, GitLab CI, GitHub Actions)
• version control server — Git/SVN hosting (GitHub, GitLab, Bitbucket)
• code quality server   — static analysis tools (SonarQube, Codecov)
• issue tracking server — bug/issue trackers (Jira, Bugzilla, GitHub Issues)

── Data & Storage ──
• database server       — relational/NoSQL databases (MySQL, PostgreSQL, MongoDB)
• file server           — network file sharing (SMB, NFS, WebDAV)
• file transfer server  — dedicated file transfer (managed file transfer platforms)
• ftp server            — FTP/SFTP services (port 21/22)
• backup server         — backup and disaster recovery systems
• search engine crawler — RARELY used. Bot/crawler infrastructure should be classified as "web server" in CTEM.
• web crawler           — RARELY used. Same as above.

── Monitoring & Logging ──
• monitoring server     — infrastructure/application monitoring (Nagios, Prometheus, Grafana, "monitor" in domain)
• logging server        — centralized log collection (ELK, Splunk, syslog)

── Specialized ──
• device management server — MDM, device fleet management
• coordination service  — distributed coordination (ZooKeeper, etcd, Consul)
• remote access server  — RDP, SSH bastion, remote desktop gateways
• print server          — network printing services
• game server           — online gaming infrastructure
• virtualization server — hypervisors, VM management (VMware, Proxmox)
"""


# ── Web Server Subsumption List ───────────────────────────────────────────────
# Categories that are specializations of "web server". If MLP predicts
# "web server" and the true category is one of these, it counts as CORRECT.

WEB_SERVER_SUBSUMES = [
    "content delivery network",
    "api endpoint",
    "certificate validation server",
    "software update server",
    "content management system",
    "application server",
    "monitoring server",
    "continuous integration server",
]


# ── System Prompt ─────────────────────────────────────────────────────────────

CLASSIFY_SYSTEM_PROMPT = (
    'You are an expert at classifying network assets for a Continuous Threat Exposure Management (CTEM) platform.\n'
    '\n'
    'CONTEXT: An MLP neural network classifier was trained on 100K+ labeled assets to predict asset categories from:\n'
    '- Domain name patterns (e.g., "api.example.com" -> api endpoint)\n'
    '- Open ports (e.g., port 25 -> mail server)\n'
    '- HTTP banners/headers (e.g., "Server: nginx" -> web server)\n'
    '- Detected technologies (e.g., WordPress -> CMS)\n'
    '- DNS query volume\n'
    '\n'
    'Your role is to independently classify the asset given the available evidence.\n'
    '\n'
    'CRITICAL RULES:\n'
    '1. You MUST only use categories from the valid list - never invent new ones.\n'
    '2. "web server" is the CORRECT classification for ANY asset with HTTP/HTTPS exposure, regardless of internal purpose.\n'
    '   Bot/crawler hosts, CDN nodes, API hosts, monitoring dashboards = all "web server" in CTEM.\n'
    '3. DO NOT override "web server" with "search engine crawler" or "web crawler".\n'
    '4. When information is sparse, "web server" as the only guess is acceptable.\n'
    '5. NEVER output "unknown" as a category. Default to "web server" when evidence is sparse.\n'
    '6. An asset can belong to MULTIPLE categories simultaneously\n'
    '   (e.g. "web server" + "mail server" if it serves HTTP and SMTP).\n'
    '\n'
    'CATEGORY HIERARCHY - "web server" SUBSUMES many specializations:\n'
    '   The following are REFINEMENTS of "web server". If the asset is one of\n'
    '   these, it should ALSO be classified as "web server":\n'
    '   - content delivery network (CDN is a type of web server)\n'
    '   - api endpoint (APIs are served over HTTP)\n'
    '   - certificate validation server (OCSP/CRL over HTTP)\n'
    '   - software update server (updates delivered over HTTP)\n'
    '   - content management system (CMS runs on a web server)\n'
    '   - application server (app servers serve HTTP)\n'
    '   - monitoring server (dashboards are web-based)\n'
    '   - continuous integration server (CI dashboards are web-based)\n'
    '\n'
    + CATEGORY_DEFINITIONS
)


# ── Classification Prompt Template ────────────────────────────────────────────
# Placeholders: {domain}, {ports}, {technologies}, {banner}, {dns_count},
#               {valid_categories}
#
# NOTE: Only the 5 asset fields are sent. NO MLP predictions go to the LLM.
#       The LLM classifies independently; comparison happens in Python.

CLASSIFY_PROMPT_TEMPLATE = (
    'Classify this network asset. Respond with ONLY a JSON object.\n'
    '\n'
    'ASSET:\n'
    '- Domain: {domain}\n'
    '- Open Ports: {ports}\n'
    '- Technologies: {technologies}\n'
    '- HTTP Banner (first 300 chars): {banner}\n'
    '- DNS Query Count: {dns_count}\n'
    '\n'
    'VALID CATEGORIES: {valid_categories}\n'
    '\n'
    'KEY RULES:\n'
    '- NEVER output "unknown". Default to "web server" if unsure.\n'
    '- An asset can have multiple categories.\n'
    '- Only use categories from the VALID CATEGORIES list.\n'
    '\n'
    'Respond with ONLY this JSON (no markdown fences, no extra text):\n'
    '{{"your_categories": ["cat1", "cat2"], '
    '"confidence": 0.85, '
    '"confidence_reasoning": "why this confidence level", '
    '"reasoning": "brief classification explanation"}}'
)
