#!/usr/bin/env node

/**
 * Word count tracking script for academic book project
 * Ensures content meets 5000-7000 word requirement (excluding references)
 */

const fs = require('fs');
const path = require('path');

// Directories to scan for word counting
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
 * Count words in a text string, excluding markdown syntax
 * @param {string} text - Text to count words in
 * @returns {number} Word count
 */
function countWords(text) {
  if (!text) return 0;

  // Remove markdown formatting and count words
  const cleanText = text
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/`[^`]*`/g, '') // Remove inline code
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1') // Remove link text
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, '') // Remove image alt text
    .replace(/[*_~`#-]/g, ' ') // Remove markdown formatting
    .replace(/\s+/g, ' ') // Replace multiple spaces with single space
    .trim();

  return cleanText.split(/\s+/).filter(word => word.length > 0).length;
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
 * Calculate total word count for the book
 * @returns {Object} Results object with counts and validation
 */
function calculateWordCount() {
  let totalWords = 0;
  const fileCounts = [];

  // Process directories
  for (const dir of SCAN_DIRS) {
    const files = getMarkdownFiles(dir);
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const wordCount = countWords(content);
      totalWords += wordCount;
      fileCounts.push({ file, wordCount });
    }
  }

  // Process additional files
  for (const file of ADDITIONAL_FILES) {
    if (fs.existsSync(file)) {
      const content = fs.readFileSync(file, 'utf8');
      const wordCount = countWords(content);
      totalWords += wordCount;
      fileCounts.push({ file, wordCount });
    }
  }

  const isValid = totalWords >= 5000 && totalWords <= 7000;

  return {
    totalWords,
    fileCounts,
    isValid,
    minRequired: 5000,
    maxRequired: 7000
  };
}

// Run the word count
const result = calculateWordCount();

console.log('Word Count Report');
console.log('=================');
console.log(`Total words: ${result.totalWords}`);
console.log(`Required range: ${result.minRequired}-${result.maxRequired} words`);
console.log(`Status: ${result.isValid ? '✅ VALID' : '❌ INVALID'}`);

if (!result.isValid) {
  if (result.totalWords < result.minRequired) {
    console.log(`⚠️  Below minimum by ${result.minRequired - result.totalWords} words`);
  } else {
    console.log(`⚠️  Above maximum by ${result.totalWords - result.maxRequired} words`);
  }
}

console.log('\nFile breakdown:');
for (const { file, wordCount } of result.fileCounts) {
  console.log(`  ${file}: ${wordCount} words`);
}

// Exit with error code if validation fails
if (!result.isValid) {
  process.exit(1);
}