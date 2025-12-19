---
id: 1
title: constitution-update
stage: constitution
date: 2025-12-17
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: Malaikaali2
command: /sp.constitution
labels: [constitution,academic,verification]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

AI / Spec-Driven Book Creation using Spec-Kit Plus
Project Objective

The objective of this project is to author a high-quality, academically rigorous book using Spec-Kit Plus and Claude Code, authored and structured through Docusaurus, and deployed via GitHub Pages. The book must meet formal academic standards while remaining clear and accessible to readers with a computer science background.

The constitution defines the immutable principles, standards, constraints, and success criteria that govern all content generation, verification, and publication activities throughout the project lifecycle.

Core Principles

Accuracy Through Primary Source Verification
All factual statements, technical explanations, statistics, and claims must be verified against authoritative primary sources. Secondary sources may be used only for contextual framing and must not replace original research or peer-reviewed material.

Academic Clarity and Precision
Content must be written for an audience with a computer science or technical background, using precise terminology, unambiguous explanations, and logically structured arguments. Concepts should be explained clearly without oversimplification.

Reproducibility and Traceability
Every non-trivial claim must be traceable to a cited source. Readers should be able to independently verify all assertions using the provided references. Citations must be embedded directly within the content.

Scholarly Rigor
Preference must be given to peer-reviewed academic literature, official standards, and reputable institutional publications. Informal or non-academic sources must be used sparingly and only when peer-reviewed material is unavailable.

Key Standards
Source Verification

All factual claims must be explicitly supported by cited sources.

Claims without citations are not permitted.

Contradictory sources must be resolved or clearly acknowledged.

Citation Requirements

Citation style: APA (American Psychological Association)

Citations must be embedded inline and listed in a references section.

All sources must be accessible and verifiable.

Source Quality

A minimum of 50% of all sources must be peer-reviewed (journals, conference papers, academic publishers).

Remaining sources may include:

Official documentation

Standards bodies

Reputable technical organizations

Plagiarism Policy

Zero tolerance for plagiarism

All content must be original, properly paraphrased, and cited.

A plagiarism check must be performed prior to submission or publication.

Writing Quality

Writing clarity must meet a Flesch-Kincaid grade level of 10–12.

Sentences should be concise, technically accurate, and logically connected.

Redundant or filler content is prohibited.

Structural and Content Constraints

Total Word Count: 5,000 – 7,000 words

Minimum Number of Sources: 15

Formatting Output:

Primary authoring in Docusaurus (Markdown)

Final export as PDF with embedded citations

Deployment Target: GitHub Pages

AI Usage Rules

Claude Code may be used for drafting, structuring, summarizing, and refining content.

AI-generated text must always be reviewed for:

Factual accuracy

Citation completeness

Originality

AI must not fabricate sources, citations, or experimental results.

Success Criteria

The project is considered successful only if all of the following conditions are met:

Every factual claim is verified against credible sources.

All citations conform to APA standards.

At least 50% of sources are peer-reviewed.

Zero plagiarism is detected in final checks.

The content passes a formal fact-checking review.

The final document adheres to word count, formatting, and clarity requirements.

The book is successfully deployed via Docusaurus on GitHub Pages.

## Response snapshot

Updated the project constitution to reflect the academic book creation requirements. The new constitution includes core principles of accuracy through primary source verification, academic clarity and precision, reproducibility and traceability, and scholarly rigor. It also establishes standards for source verification, citation requirements (APA style), source quality (50% peer-reviewed minimum), plagiarism policy, writing quality standards, structural constraints, and AI usage rules. The governance section outlines compliance requirements and amendment procedures.

## Outcome

- ✅ Impact: Updated project constitution with academic standards and verification requirements
- 🧪 Tests: Not applicable
- 📁 Files: Updated .specify/memory/constitution.md with new academic-focused principles
- 🔁 Next prompts: Proceed with creating book content following the established constitution
- 🧠 Reflection: The constitution now properly establishes academic rigor requirements for the book project

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): Not applicable
- Prompt variant (if applicable): Not applicable
- Next experiment (smallest change to try): Begin creating book content based on constitution