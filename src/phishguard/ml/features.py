import re
from urllib.parse import urlparse

import numpy as np

SUSPICIOUS_KEYWORDS = (
    "login", "signin", "verify", "update", "secure",
    "account", "bank", "password", "confirm"
)

IPV4_REGEX = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")

#Return 1 if hostname looks like an IPv4 address, else return 0
def _is_ipv4(hostname: str) -> int:

    if not hostname:
        return 0
    return int(bool(IPV4_REGEX.match(hostname)))

#Convert URL string into a numeric feature vector
def extract_features(url:str) -> np.ndarray:
    u = (url or "").strip()

    #Extract url info
    parsed = urlparse(u)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    query = (parsed.query or "").lower()
    full_lower = u.lower()

    #Count features:
    #Phishing url usually have unusual structure:
    length = len(u)
    dot_count = u.count(".")
    dash_count = u.count("-")
    slash_count = u.count("/")
    digit_count = sum(ch.isdigit() for ch in u)

    #Pattern features:
    #@ is sus since it can hide real domain
    has_at = int("@" in u)
    #https is a weak positive signal
    has_https = int(parsed.scheme.lower() == "https")
    #IP address hostnames are often sus
    has_ip = _is_ipv4(host)

    #Subdomain depth
    subdomain_dots = host.count(".")

    #Keyword count
    keyword_count = sum(kw in full_lower for kw in SUSPICIOUS_KEYWORDS)

    #Query length
    query_length = len(query)

    #Packaging
    feats = np.array(
        [
            length,
            dot_count,
            dash_count,
            slash_count,
            digit_count,
            has_at,
            has_https,
            has_ip,
            subdomain_dots,
            keyword_count,
            query_length,
        ],
        dtype=np.float32,
    )
    return feats




