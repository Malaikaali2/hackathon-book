/**
 * Source tracking database for academic book project
 * Ensures ≥50% peer-reviewed requirement and manages citations
 */

const fs = require('fs');
const path = require('path');

class SourceTracker {
  constructor(filePath = './src/data/sources.json') {
    this.filePath = filePath;
    this.sources = this.loadSources();
  }

  /**
   * Load sources from file
   * @returns {Array} Array of source objects
   */
  loadSources() {
    if (fs.existsSync(this.filePath)) {
      const data = fs.readFileSync(this.filePath, 'utf8');
      return JSON.parse(data);
    }
    return [];
  }

  /**
   * Save sources to file
   */
  saveSources() {
    // Ensure directory exists
    const dir = path.dirname(this.filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(this.filePath, JSON.stringify(this.sources, null, 2));
  }

  /**
   * Add a new source
   * @param {Object} source - Source object with required fields
   * @returns {string} ID of the added source
   */
  addSource(source) {
    // Validate required fields
    if (!source.title || !source.author || (!source.publication_date && !source.year) || !source.url) {
      throw new Error('Source must have title, author, publication date/year, and URL');
    }

    // Generate a unique ID
    const id = this.generateSourceId(source);
    const newSource = {
      id,
      ...source,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };

    this.sources.push(newSource);
    this.saveSources();
    return id;
  }

  /**
   * Update an existing source
   * @param {string} id - Source ID to update
   * @param {Object} updates - Fields to update
   */
  updateSource(id, updates) {
    const index = this.sources.findIndex(s => s.id === id);
    if (index === -1) {
      throw new Error(`Source with ID ${id} not found`);
    }

    // Update fields and timestamp
    this.sources[index] = {
      ...this.sources[index],
      ...updates,
      updated_at: new Date().toISOString()
    };

    this.saveSources();
  }

  /**
   * Get a source by ID
   * @param {string} id - Source ID
   * @returns {Object|null} Source object or null if not found
   */
  getSource(id) {
    return this.sources.find(s => s.id === id) || null;
  }

  /**
   * Get all sources
   * @returns {Array} Array of all sources
   */
  getAllSources() {
    return this.sources;
  }

  /**
   * Get peer-reviewed sources only
   * @returns {Array} Array of peer-reviewed sources
   */
  getPeerReviewedSources() {
    return this.sources.filter(s => this.isPeerReviewed(s));
  }

  /**
   * Check if a source is peer-reviewed
   * @param {Object} source - Source object to check
   * @returns {boolean} True if source is peer-reviewed
   */
  isPeerReviewed(source) {
    if (!source) return false;

    // Check various possible properties that indicate peer review
    return source.is_peer_reviewed === true ||
           source.type === 'peer-reviewed' ||
           (source.type && source.type.toLowerCase().includes('peer')) ||
           (source.journal && source.journal.toLowerCase().includes('journal'));
  }

  /**
   * Count peer-reviewed sources
   * @returns {number} Count of peer-reviewed sources
   */
  countPeerReviewed() {
    return this.getPeerReviewedSources().length;
  }

  /**
   * Calculate percentage of peer-reviewed sources
   * @returns {number} Percentage of peer-reviewed sources (0-100)
   */
  calculatePeerReviewedPercentage() {
    if (this.sources.length === 0) return 0;
    return Math.round((this.countPeerReviewed() / this.sources.length) * 100);
  }

  /**
   * Validate that peer-reviewed requirement is met (≥50%)
   * @returns {Object} Validation result with status and details
   */
  validatePeerReviewedRequirement() {
    const peerReviewedCount = this.countPeerReviewed();
    const totalCount = this.sources.length;
    const percentage = totalCount > 0 ? (peerReviewedCount / totalCount) * 100 : 0;
    const isValid = percentage >= 50;

    return {
      isValid,
      peerReviewedCount,
      totalCount,
      percentage: percentage.toFixed(2),
      message: isValid
        ? `✅ ${percentage.toFixed(2)}% peer-reviewed sources (${peerReviewedCount}/${totalCount}) - Requirement met`
        : `❌ ${percentage.toFixed(2)}% peer-reviewed sources (${peerReviewedCount}/${totalCount}) - Requirement not met (need ≥50%)`
    };
  }

  /**
   * Generate a unique ID for a source
   * @param {Object} source - Source object to generate ID for
   * @returns {string} Unique ID
   */
  generateSourceId(source) {
    // Create a simple ID based on title and author
    const titleSlug = source.title.toLowerCase().replace(/[^a-z0-9]/g, '-').substring(0, 30);
    const authorSlug = (Array.isArray(source.author) ? source.author[0] : source.author)
      .toLowerCase().replace(/[^a-z0-9]/g, '-').substring(0, 20);

    const timestamp = Date.now();
    return `${titleSlug}_${authorSlug}_${timestamp}`;
  }

  /**
   * Find sources by topic area
   * @param {string} topic - Topic to search for
   * @returns {Array} Array of matching sources
   */
  findByTopic(topic) {
    return this.sources.filter(s =>
      s.topic_areas && s.topic_areas.some(t =>
        t.toLowerCase().includes(topic.toLowerCase())
      )
    );
  }

  /**
   * Export sources in APA format
   * @returns {Array} Array of APA formatted citations
   */
  exportAPACitations() {
    return this.sources.map(source => this.formatAPACitation(source));
  }

  /**
   * Format a source in APA style
   * @param {Object} source - Source object to format
   * @returns {string} Formatted APA citation
   */
  formatAPACitation(source) {
    let citation = '';

    // Format author names (Last name, First initial.)
    if (source.author) {
      if (Array.isArray(source.author)) {
        citation += source.author.map(this.formatAuthorName).join(', ');
      } else {
        citation += this.formatAuthorName(source.author);
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
  formatAuthorName(author) {
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
}

module.exports = SourceTracker;