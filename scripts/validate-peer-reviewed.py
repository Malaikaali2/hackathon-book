#!/usr/bin/env python3
"""
Script to validate that ≥50% of sources are peer-reviewed for Modules 1-2
"""

import re
from pathlib import Path

def is_peer_reviewed_reference(ref_text):
    """Determine if a reference is likely peer-reviewed based on content"""
    ref_lower = ref_text.lower()

    # Positive indicators of peer-reviewed sources
    peer_review_indicators = [
        'transactions', 'journal', 'conference', 'proceedings', 'international conference',
        'ieee', 'acm', 'springer', 'mit press', 'elsevier', 'wiley', 'nature', 'science',
        'electronics', 'autonomous robots', 'computer vision', 'robotics and automation',
        'international journal', 'research', 'systems', 'automated', 'vision', 'learning',
        'machine', 'artificial intelligence', 'pattern recognition', 'intelligent systems',
        'computer vision and image understanding', 'international conference on robotics',
        'ieee transactions', 'acm transactions', 'springer international', 'springer handbook',
        'international symposium', 'workshop', 'transactions on', 'journal of', 'computer',
        'engineering', 'automation', 'intelligent', 'robotics', 'vision and pattern',
        'ieee robotics', 'ieee/rsa', 'ieee/rsj', 'aiaa', 'asme', 'siam', 'mit', 'cambridge',
        'oxford', 'elsevier', 'springer-verlag', 'association for', 'acm digital library',
        'electronic', 'robotics', 'automated', 'intelligent systems', 'pattern recognition',
        'computer vision and image understanding', 'international journal of robotics',
        'ieee robotics & automation magazine', 'autonomous robots', 'computer vision',
        'artificial intelligence', 'ieee transactions on', 'acm transactions on'
    ]

    # Negative indicators (likely not peer-reviewed)
    non_peer_indicators = [
        'documentation', 'manual', 'guide', 'reference', 'website', 'retrieved from', 'accessed',
        'from http', 'from https', 'online', 'personal communication', 'blog', 'tutorial',
        'developer', 'user manual', 'software', 'api', 'library', 'package', 'tool',
        'documentation', 'help', 'support', 'faq', 'instructions', 'retrieved', 'from', 'accessed on',
        'available', 'at', 'website', 'visited', 'last accessed', 'home page', 'online documentation'
    ]

    # Check for peer review indicators
    has_peer_indicator = any(indicator in ref_lower for indicator in peer_review_indicators)

    # Check for non-peer indicators
    has_non_peer_indicator = any(indicator in ref_lower for indicator in non_peer_indicators)

    # Academic publishers are generally peer-reviewed
    academic_publishers = ['mit press', 'springer', 'wiley', 'elsevier', 'cambridge university press',
                          'oxford university press', 'ieee press', 'aiaa', 'asme', 'siam', 'springer international']
    is_academic_book = any(publisher in ref_lower for publisher in academic_publishers)

    # Papers in conferences and journals are peer-reviewed
    is_journal_paper = ('transactions' in ref_lower or 'journal' in ref_lower or
                       'conference' in ref_lower or 'proceedings' in ref_lower or
                       'international conference' in ref_lower or 'symposium' in ref_lower)

    # Determine if peer-reviewed
    if (has_peer_indicator and not has_non_peer_indicator) or is_academic_book or is_journal_paper:
        return True
    elif has_non_peer_indicator and not has_peer_indicator:
        return False
    else:
        # Default to being conservative - if uncertain, consider not peer-reviewed
        return False

def count_peer_reviewed_sources(ref_file_path):
    """Count peer-reviewed vs non-peer-reviewed sources in the references file"""
    with open(ref_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content into lines
    lines = content.split('\n')

    # Count references by type
    total_refs = 0
    peer_reviewed_count = 0
    current_ref = ""

    for line in lines:
        line = line.strip()

        # Look for reference lines (start with number followed by period)
        ref_match = re.match(r'^(\d+)\.\s+(.+)', line)
        if ref_match:
            # If we have a previous reference to analyze, process it
            if current_ref:
                total_refs += 1
                if is_peer_reviewed_reference(current_ref):
                    peer_reviewed_count += 1
                    print(f"✓ Peer-reviewed: {current_ref[:100]}...")
                else:
                    print(f"  Non-peer-reviewed: {current_ref[:100]}...")

            # Start processing the new reference
            current_ref = ref_match.group(2)
        elif current_ref and (line.startswith(' ') or line.startswith('\t')) and line.strip() != "":
            # Continuation of the current reference
            current_ref += " " + line.strip()
        elif current_ref and line.strip() == "":
            # End of current reference (empty line), process it
            total_refs += 1
            if is_peer_reviewed_reference(current_ref):
                peer_reviewed_count += 1
                print(f"✓ Peer-reviewed: {current_ref[:100]}...")
            else:
                print(f"  Non-peer-reviewed: {current_ref[:100]}...")
            current_ref = ""

    # Process the last reference if exists
    if current_ref:
        total_refs += 1
        if is_peer_reviewed_reference(current_ref):
            peer_reviewed_count += 1
            print(f"✓ Peer-reviewed: {current_ref[:100]}...")
        else:
            print(f"  Non-peer-reviewed: {current_ref[:100]}...")

    return total_refs, peer_reviewed_count

def main():
    ref_file = "docs/references/references.md"

    if not Path(ref_file).exists():
        print(f"❌ References file not found: {ref_file}")
        return 1

    print("🔍 Analyzing reference types for peer-review validation...")
    print("="*60)

    total_refs, peer_reviewed_count = count_peer_reviewed_sources(ref_file)

    print("="*60)
    print(f"📊 Results:")
    print(f"   Total references: {total_refs}")
    print(f"   Peer-reviewed: {peer_reviewed_count}")
    print(f"   Non-peer-reviewed: {total_refs - peer_reviewed_count}")

    if total_refs > 0:
        percentage = (peer_reviewed_count / total_refs) * 100
        print(f"   Percentage peer-reviewed: {percentage:.1f}%")

        if percentage >= 50:
            print(f"✅ VALIDATION PASSED: {percentage:.1f}% ≥ 50% peer-reviewed")
            print(f"🎯 Requirement satisfied: ≥50% of sources are peer-reviewed")
            return 0
        else:
            print(f"❌ VALIDATION FAILED: {percentage:.1f}% < 50% peer-reviewed")
            print(f"⚠️  Need {(total_refs//2 + 1) - peer_reviewed_count} more peer-reviewed sources to meet 50% requirement")
            return 1
    else:
        print("❌ No references found in file")
        return 1

if __name__ == "__main__":
    exit(main())