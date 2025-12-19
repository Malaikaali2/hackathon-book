# Quickstart Guide: Physical AI & Humanoid Robotics Book Development

## Prerequisites
- Git installed
- Node.js 18+ for Docusaurus
- Python 3.8+ for development tools
- Access to academic databases for research
- GitHub account for deployment

## Environment Setup
1. Clone the repository: `git clone <repo-url>`
2. Install Docusaurus: `npm install` in the root directory
3. Install Python dependencies: `pip install -r requirements.txt` (if exists)
4. Verify Claude Code setup: `claude --version`

## Development Workflow
1. **Research Phase**: Use Claude Code to research and draft content while collecting sources
2. **Writing Phase**: Create content in Docusaurus Markdown format
3. **Validation Phase**: Check word count, citation compliance, and readability
4. **Review Phase**: Validate against all acceptance criteria

## Content Creation Process
1. Follow the 13-week roadmap structure
2. Maintain 50%+ peer-reviewed sources for each module
3. Include inline APA citations for all claims
4. Write at Flesch-Kincaid grade level 10-12
5. Implement hands-on labs for each module

## Validation Commands
- Word count: `find . -name "*.md" -exec wc -w {} +` (excluding references)
- Citation check: Manual review of all inline citations
- Readability: Use online Flesch-Kincaid tools
- Plagiarism: Use preferred detection tools before final submission

## Deployment
1. Build Docusaurus site: `npm run build`
2. Deploy to GitHub Pages: `npm run deploy`
3. Export PDF: Use Docusaurus PDF export or print to PDF functionality