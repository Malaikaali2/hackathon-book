/**
 * Citation utilities for academic book project
 * Provides functions for APA formatting and citation management
 */

/**
 * Format a citation in APA style
 * @param {Object} source - Source object with title, author, year, etc.
 * @returns {string} Formatted APA citation
 */
export function formatAPACitation(source) {
  if (!source) return '';

  let citation = '';

  // Format author names (Last name, First initial.)
  if (source.author) {
    if (Array.isArray(source.author)) {
      citation += source.author.map(formatAuthorName).join(', ');
    } else {
      citation += formatAuthorName(source.author);
    }
    citation += '. ';
  }

  // Add publication year
  if (source.year || source.publication_date) {
    const year = source.year || new Date(source.publication_date).getFullYear();
    citation += `(${year}). `;
  }

  // Add title
  if (source.title) {
    citation += `${source.title}. `;
  }

  // Add publication info if available
  if (source.journal || source.publisher) {
    citation += source.journal ? `${source.journal}. ` : '';
    citation += source.publisher ? `${source.publisher}. ` : '';
  }

  // Add URL if available
  if (source.url) {
    citation += source.url;
  }

  return citation;
}

/**
 * Format an author's name for APA style
 * @param {string|Object} author - Author name as string or object
 * @returns {string} Formatted author name
 */
function formatAuthorName(author) {
  if (typeof author === 'string') {
    // Simple case: "Smith, J."
    if (author.includes(',')) {
      return author.trim();
    }
    // If it's just a full name, try to format it
    const parts = author.trim().split(' ');
    if (parts.length >= 2) {
      const lastName = parts[parts.length - 1];
      const initials = parts.slice(0, -1).map(name => name[0] + '.').join('');
      return `${lastName}, ${initials}`;
    }
    return author;
  }

  if (typeof author === 'object') {
    // If author is an object with first/last properties
    if (author.firstName && author.lastName) {
      return `${author.lastName}, ${author.firstName.charAt(0)}.`;
    }
  }

  return author || '';
}

/**
 * Validate that a source meets academic requirements
 * @param {Object} source - Source object to validate
 * @returns {Array} List of validation errors
 */
export function validateSource(source) {
  const errors = [];

  if (!source.title) {
    errors.push('Source must have a title');
  }

  if (!source.author) {
    errors.push('Source must have an author');
  }

  if (!source.publication_date && !source.year) {
    errors.push('Source must have a publication date or year');
  }

  if (!source.type) {
    errors.push('Source must have a type (e.g., peer-reviewed, documentation)');
  }

  if (!source.url) {
    errors.push('Source must have a URL for verification');
  }

  return errors;
}

/**
 * Check if a source is peer-reviewed
 * @param {Object} source - Source object
 * @returns {boolean} True if source is peer-reviewed
 */
export function isPeerReviewed(source) {
  if (!source) return false;

  // Check various possible properties that indicate peer review
  return source.is_peer_reviewed === true ||
         source.type === 'peer-reviewed' ||
         (source.type && source.type.toLowerCase().includes('peer')) ||
         (source.journal && source.journal.toLowerCase().includes('journal'));
}

/**
 * Count peer-reviewed sources in a list
 * @param {Array} sources - Array of source objects
 * @returns {number} Count of peer-reviewed sources
 */
export function countPeerReviewedSources(sources) {
  if (!Array.isArray(sources)) return 0;
  return sources.filter(isPeerReviewed).length;
}

/**
 * Calculate percentage of peer-reviewed sources
 * @param {Array} sources - Array of source objects
 * @returns {number} Percentage of peer-reviewed sources (0-100)
 */
export function calculatePeerReviewedPercentage(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return 0;
  const peerReviewedCount = countPeerReviewedSources(sources);
  return Math.round((peerReviewedCount / sources.length) * 100);
}