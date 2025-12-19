#!/usr/bin/env node

/**
 * Plagiarism detection workflow for academic book project
 * Implements zero tolerance policy for plagiarism
 */

const fs = require('fs');
const path = require('path');

// Directories to scan for plagiarism check
const SCAN_DIRS = [
  'docs/intro',
  'docs/module-1-ros',
  'docs/module-2-digital-twin',
  'docs/module-3-ai-brain',
  'docs/module-4-vla',
  'docs/capstone',
  'docs/appendices'
];

// Additional files to include
const ADDITIONAL_FILES = [
  'docs/intro.md'
];

/**
 * Check for potential plagiarism by looking for common patterns
 * This is a basic implementation - in a real scenario, you'd integrate with
 * external plagiarism detection services
 * @param {string} text - Text to check for plagiarism
 * @returns {Array} Array of potential issues found
 */
function checkForPlagiarism(text) {
  const issues = [];

  if (!text) return issues;

  // Check for excessive consecutive copied text (more than 10 words the same)
  const words = text.toLowerCase().split(/\s+/).filter(w => w.length > 0);
  const commonPhrases = [
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'
  ];

  // Look for long sequences of identical words (potential direct copying)
  // This is a simplified check - real implementation would need more sophisticated algorithms
  const longSequences = findLongSequences(words, 15); // 15+ consecutive identical words

  if (longSequences.length > 0) {
    issues.push({
      type: 'potential_direct_copy',
      description: 'Found long sequences of identical words that may indicate direct copying',
      sequences: longSequences
    });
  }

  // Check for proper citation format (this is basic - real implementation would be more thorough)
  const citationCheck = checkCitations(text);
  if (!citationCheck.hasProperCitations) {
    issues.push({
      type: 'missing_citations',
      description: 'Text may contain uncited content. Ensure all non-original content is properly cited.',
      suggestions: citationCheck.suggestions
    });
  }

  return issues;
}

/**
 * Find long sequences of identical words
 * @param {Array} words - Array of words
 * @param {number} minLength - Minimum length of sequence to consider
 * @returns {Array} Array of long sequences found
 */
function findLongSequences(words, minLength) {
  const sequences = [];
  let currentSequence = [];

  for (let i = 0; i < words.length; i++) {
    if (currentSequence.length === 0) {
      currentSequence.push(words[i]);
    } else if (words[i] === currentSequence[currentSequence.length - 1]) {
      currentSequence.push(words[i]);
    } else {
      if (currentSequence.length >= minLength) {
        sequences.push({
          text: currentSequence.join(' '),
          length: currentSequence.length,
          position: i - currentSequence.length
        });
      }
      currentSequence = [words[i]];
    }
  }

  // Check the last sequence
  if (currentSequence.length >= minLength) {
    sequences.push({
      text: currentSequence.join(' '),
      length: currentSequence.length,
      position: words.length - currentSequence.length
    });
  }

  return sequences;
}

/**
 * Check for proper citation format
 * @param {string} text - Text to check
 * @returns {Object} Citation check results
 */
function checkCitations(text) {
  // Look for citation patterns (this is a simplified check)
  const hasParentheticalCitations = /(\([A-Z][a-z]+, \d{4}\)|\([A-Z][a-z]+ et al\., \d{4}\))/g.test(text);
  const hasBracketCitations = /\[\d+\]/g.test(text);
  const hasInlineCitations = /According to [A-Z][a-z]+.*?\d{4}.*?:|As [A-Z][a-z]+.*?\d{4}.*?stated/gi.test(text);

  const hasProperCitations = hasParentheticalCitations || hasBracketCitations || hasInlineCitations;

  const suggestions = [];
  if (!hasProperCitations) {
    suggestions.push('Add proper APA citations for any non-original content');
    suggestions.push('Use (Author, Year) format for in-text citations');
  }

  return {
    hasProperCitations,
    suggestions
  };
}

/**
 * Get all markdown files in a directory recursively
 * @param {string} dir - Directory to scan
 * @returns {string[]} Array of markdown file paths
 */
function getMarkdownFiles(dir) {
  if (!fs.existsSync(dir)) return [];

  const files = [];
  const items = fs.readdirSync(dir);

  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      files.push(...getMarkdownFiles(fullPath));
    } else if (item.endsWith('.md')) {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Run plagiarism check on the entire book
 * @returns {Object} Results object with plagiarism check results
 */
function runPlagiarismCheck() {
  let totalIssues = 0;
  const fileIssues = [];

  // Process directories
  for (const dir of SCAN_DIRS) {
    const files = getMarkdownFiles(dir);
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const issues = checkForPlagiarism(content);

      if (issues.length > 0) {
        totalIssues += issues.length;
        fileIssues.push({ file, issues });
      }
    }
  }

  // Process additional files
  for (const file of ADDITIONAL_FILES) {
    if (fs.existsSync(file)) {
      const content = fs.readFileSync(file, 'utf8');
      const issues = checkForPlagiarism(content);

      if (issues.length > 0) {
        totalIssues += issues.length;
        fileIssues.push({ file, issues });
      }
    }
  }

  const isValid = totalIssues === 0;

  return {
    isValid,
    totalIssues,
    fileIssues,
    policy: 'Zero tolerance for plagiarism - all content must be original with proper attribution'
  };
}

// Run the plagiarism check
const result = runPlagiarismCheck();

console.log('Plagiarism Detection Report');
console.log('===========================');
console.log(`Policy: ${result.policy}`);
console.log(`Total issues found: ${result.totalIssues}`);
console.log(`Status: ${result.isValid ? '✅ VALID (No plagiarism detected)' : '❌ INVALID (Issues found)'}`);

if (result.fileIssues.length > 0) {
  console.log('\nIssues by file:');
  for (const { file, issues } of result.fileIssues) {
    console.log(`\n  ${file}:`);
    for (const issue of issues) {
      console.log(`    - ${issue.type}: ${issue.description}`);
      if (issue.sequences) {
        for (const seq of issue.sequences) {
          console.log(`      Sequence: "${seq.text.substring(0, 50)}..." (${seq.length} words)`);
        }
      }
      if (issue.suggestions) {
        for (const suggestion of issue.suggestions) {
          console.log(`      Suggestion: ${suggestion}`);
        }
      }
    }
  }
} else {
  console.log('\nNo plagiarism issues detected.');
}

// Exit with error code if validation fails
if (!result.isValid) {
  console.log('\n⚠️  Plagiarism check failed. All content must be original with proper citations.');
  process.exit(1);
}