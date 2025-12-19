#!/usr/bin/env python3
"""
Reference verification script for Physical AI & Humanoid Robotics book
Checks that all cited sources are accessible and verifiable
"""

import re
import requests
import time
from urllib.parse import urlparse
import sys
from pathlib import Path

def extract_citations_from_md(file_path):
    """Extract all citations from a markdown file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern to match citations like (Author et al., Year)
    citation_pattern = r'\([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s\-,]+et al\.,\s*\d{4}\)'
    citations = re.findall(citation_pattern, content)

    # Also look for direct URL citations
    url_pattern = r'https?://[^\s\)<"\'\n\r]+'
    urls = re.findall(url_pattern, content)

    return citations, urls

def verify_url_accessibility(url, timeout=10):
    """Verify that a URL is accessible"""
    try:
        # Some URLs might be DOI links or other identifiers
        parsed = urlparse(url)

        # Handle special cases
        if 'doi.org' in url or 'dx.doi.org' in url:
            # DOI links often redirect, allow longer timeout
            response = requests.head(url, timeout=timeout, allow_redirects=True)
        else:
            response = requests.head(url, timeout=timeout, allow_redirects=True)

        # Consider it accessible if we get a successful response or redirect
        return response.status_code < 400
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Error accessing {url}: {str(e)}")
        return False
    except Exception as e:
        print(f"  ❌ Unexpected error accessing {url}: {str(e)}")
        return False

def verify_citation_format(citation):
    """Verify that a citation follows APA format"""
    # Basic check for APA format: (Author et al., Year)
    pattern = r'^\([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ\s\-,]+et al\.,\s*\d{4}\)$'
    return bool(re.match(pattern, citation.strip()))

def main():
    print("🔍 Starting reference verification for Physical AI & Humanoid Robotics book...")

    # Define the files to check
    md_files = [
        "docs/intro.md",
        "docs/module-1-ros/intro.md",
        "docs/module-1-ros/fundamentals.md",
        "docs/module-1-ros/lab-publisher-subscriber.md",
        "docs/module-2-digital-twin/intro.md",
        "docs/module-2-digital-twin/gazebo-fundamentals.md",
        "docs/module-2-digital-twin/custom-environment.md",
        "docs/module-2-digital-twin/sensor-simulation.md",
        "docs/module-2-digital-twin/unity-simulation.md",
        "docs/module-2-digital-twin/sim-to-reality.md",
        "docs/module-2-digital-twin/lab-gazebo-world.md",
        "docs/module-2-digital-twin/summary.md"
    ]

    all_citations = set()
    all_urls = set()

    # Extract citations and URLs from all files
    for file_path in md_files:
        if Path(file_path).exists():
            print(f"\n📄 Checking {file_path}...")
            citations, urls = extract_citations_from_md(file_path)

            for citation in citations:
                all_citations.add(citation)
                print(f"  Found citation: {citation}")

            for url in urls:
                all_urls.add(url)
                print(f"  Found URL: {url}")
        else:
            print(f"⚠️  File not found: {file_path}")

    print(f"\n📊 Summary:")
    print(f"- Total unique citations found: {len(all_citations)}")
    print(f"- Total unique URLs found: {len(all_urls)}")

    # Verify citation formats
    print(f"\n🔍 Verifying citation formats...")
    valid_citations = []
    invalid_citations = []

    for citation in all_citations:
        if verify_citation_format(citation):
            valid_citations.append(citation)
            print(f"  ✅ Valid: {citation}")
        else:
            invalid_citations.append(citation)
            print(f"  ❌ Invalid: {citation}")

    # Verify URL accessibility
    print(f"\n🌐 Verifying URL accessibility...")
    accessible_urls = []
    inaccessible_urls = []

    for i, url in enumerate(all_urls, 1):
        print(f"  Checking URL {i}/{len(all_urls)}: {url[:50]}{'...' if len(url) > 50 else ''}")
        if verify_url_accessibility(url):
            accessible_urls.append(url)
            print(f"    ✅ Accessible")
        else:
            inaccessible_urls.append(url)
            print(f"    ❌ Inaccessible")

        # Be respectful to servers
        time.sleep(0.5)

    # Final report
    print(f"\n📋 VERIFICATION REPORT")
    print(f"========================")
    print(f"Citations - Valid: {len(valid_citations)}, Invalid: {len(invalid_citations)}")
    print(f"URLs - Accessible: {len(accessible_urls)}, Inaccessible: {len(inaccessible_urls)}")

    if invalid_citations:
        print(f"\n❌ Invalid citations found:")
        for cit in invalid_citations:
            print(f"  - {cit}")

    if inaccessible_urls:
        print(f"\n❌ Inaccessible URLs found:")
        for url in inaccessible_urls:
            print(f"  - {url}")

    # Success criteria
    total_citations = len(valid_citations) + len(invalid_citations)
    total_urls = len(accessible_urls) + len(inaccessible_urls)

    citations_valid = len(invalid_citations) == 0
    urls_accessible = len(inaccessible_urls) == 0

    print(f"\n🎯 VERIFICATION RESULTS")
    print(f"=====================")
    print(f"All citations properly formatted: {'✅ YES' if citations_valid else '❌ NO'}")
    print(f"All URLs accessible: {'✅ YES' if urls_accessible else '❌ NO'}")

    if citations_valid and urls_accessible:
        print(f"\n🎉 All references verified successfully!")
        print(f"All citations follow APA format and all URLs are accessible.")
        return 0
    else:
        print(f"\n⚠️  Some references need attention:")
        if not citations_valid:
            print(f"  - Fix {len(invalid_citations)} citation format issues")
        if not urls_accessible:
            print(f"  - Fix {len(inaccessible_urls)} inaccessible URLs")
        return 1

if __name__ == "__main__":
    sys.exit(main())