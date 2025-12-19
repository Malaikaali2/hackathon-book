#!/usr/bin/env node

/**
 * Readability analysis script for academic book project
 * Validates Flesch-Kincaid grade level 10-12 compliance
 */

const fs = require('fs');
const path = require('path');

// Directories to scan for readability analysis
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
 * Calculate Flesch-Kincaid Grade Level
 * Formula: 0.39 * (total words / total sentences) + 11.8 * (total syllables / total words) - 15.59
 * @param {string} text - Text to analyze
 * @returns {number} Flesch-Kincaid Grade Level
 */
function calculateFleschKincaidGradeLevel(text) {
  if (!text) return 0;

  // Count sentences (periods, exclamation marks, question marks)
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0).length || 1;

  // Count words
  const words = text.split(/\s+/).filter(w => w.length > 0).length || 1;

  // Count syllables
  const syllables = countSyllables(text);

  // Calculate Flesch-Kincaid Grade Level
  const gradeLevel = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59;

  return Math.max(0, gradeLevel); // Don't return negative values
}

/**
 * Count syllables in text (simplified algorithm)
 * @param {string} text - Text to analyze
 * @returns {number} Number of syllables
 */
function countSyllables(text) {
  // Remove punctuation and convert to lowercase
  const cleanText = text.toLowerCase().replace(/[^a-z\s]/g, ' ');

  let syllableCount = 0;
  const words = cleanText.split(/\s+/).filter(w => w.length > 0);

  for (const word of words) {
    syllableCount += countSyllablesInWord(word);
  }

  return syllableCount;
}

/**
 * Count syllables in a single word
 * @param {string} word - Word to analyze
 * @returns {number} Number of syllables in the word
 */
function countSyllablesInWord(word) {
  if (!word) return 0;

  // Count vowel groups
  const vowelGroups = word.match(/[aeiouy]+/g);
  let count = vowelGroups ? vowelGroups.length : 0;

  // Subtract 1 for silent 'e' at the end
  if (word.endsWith('e') && count > 1) {
    count--;
  }

  // Add 1 if word ends with 'le' preceded by a consonant
  if (word.endsWith('le') && word.length > 2 && !'aeiouy'.includes(word[word.length - 3])) {
    count++;
  }

  // Ensure at least 1 syllable for any word
  return Math.max(1, count);
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
 * Calculate readability for the entire book
 * @returns {Object} Results object with readability scores
 */
function calculateReadability() {
  let totalWords = 0;
  let totalSentences = 0;
  let totalSyllables = 0;
  const fileReadabilities = [];

  // Process directories
  for (const dir of SCAN_DIRS) {
    const files = getMarkdownFiles(dir);
    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const gradeLevel = calculateFleschKincaidGradeLevel(content);

      // Extract basic stats for this file
      const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0).length || 1;
      const words = content.split(/\s+/).filter(w => w.length > 0).length || 1;
      const syllables = countSyllables(content);

      totalWords += words;
      totalSentences += sentences;
      totalSyllables += syllables;

      fileReadabilities.push({ file, gradeLevel, words, sentences, syllables });
    }
  }

  // Process additional files
  for (const file of ADDITIONAL_FILES) {
    if (fs.existsSync(file)) {
      const content = fs.readFileSync(file, 'utf8');
      const gradeLevel = calculateFleschKincaidGradeLevel(content);

      // Extract basic stats for this file
      const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0).length || 1;
      const words = content.split(/\s+/).filter(w => w.length > 0).length || 1;
      const syllables = countSyllables(content);

      totalWords += words;
      totalSentences += sentences;
      totalSyllables += syllables;

      fileReadabilities.push({ file, gradeLevel, words, sentences, syllables });
    }
  }

  // Calculate overall grade level
  const overallGradeLevel = totalSentences > 0 && totalWords > 0
    ? 0.39 * (totalWords / totalSentences) + 11.8 * (totalSyllables / totalWords) - 15.59
    : 0;

  const isValid = overallGradeLevel >= 10 && overallGradeLevel <= 12;

  return {
    overallGradeLevel: Math.max(0, overallGradeLevel),
    fileReadabilities,
    isValid,
    minRequired: 10,
    maxRequired: 12
  };
}

// Run the readability analysis
const result = calculateReadability();

console.log('Readability Report (Flesch-Kincaid Grade Level)');
console.log('===============================================');
console.log(`Overall Grade Level: ${result.overallGradeLevel.toFixed(2)}`);
console.log(`Required range: ${result.minRequired}-${result.maxRequired} grade level`);
console.log(`Status: ${result.isValid ? '✅ VALID' : '❌ INVALID'}`);

if (!result.isValid) {
  if (result.overallGradeLevel < result.minRequired) {
    console.log(`⚠️  Below minimum by ${result.minRequired - result.overallGradeLevel.toFixed(2)} grade levels`);
  } else {
    console.log(`⚠️  Above maximum by ${result.overallGradeLevel.toFixed(2) - result.maxRequired} grade levels`);
  }
}

console.log('\nFile breakdown:');
for (const { file, gradeLevel } of result.fileReadabilities) {
  console.log(`  ${file}: ${gradeLevel.toFixed(2)} grade level`);
}

// Exit with error code if validation fails
if (!result.isValid) {
  process.exit(1);
}