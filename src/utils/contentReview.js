/**
 * Content review checklist based on constitution requirements
 * Validates content against academic standards and constitutional principles
 */

const fs = require('fs');
const path = require('path');

class ContentReview {
  constructor() {
    this.constitutionRequirements = {
      accuracy: {
        name: 'Accuracy Through Primary Source Verification',
        description: 'All factual statements, technical explanations, statistics, and claims must be verified against authoritative primary sources',
        checkFunction: this.checkPrimarySourceVerification
      },
      clarity: {
        name: 'Academic Clarity and Precision',
        description: 'Content must be written for an audience with a computer science or technical background, using precise terminology, unambiguous explanations, and logically structured arguments',
        checkFunction: this.checkAcademicClarity
      },
      reproducibility: {
        name: 'Reproducibility and Traceability',
        description: 'Every non-trivial claim must be traceable to a cited source. Readers should be able to independently verify all assertions using the provided references',
        checkFunction: this.checkReproducibility
      },
      rigor: {
        name: 'Scholarly Rigor',
        description: 'Preference must be given to peer-reviewed academic literature, official standards, and reputable institutional publications',
        checkFunction: this.checkScholarlyRigor
      },
      sourceVerification: {
        name: 'Source Verification',
        description: 'All factual claims must be explicitly supported by cited sources. Claims without citations are not permitted',
        checkFunction: this.checkSourceVerification
      },
      citationRequirements: {
        name: 'Citation Requirements',
        description: 'Citation style must follow APA (American Psychological Association) standards. Citations must be embedded inline and listed in a references section',
        checkFunction: this.checkCitationRequirements
      }
    };
  }

  /**
   * Check primary source verification
   * @param {string} content - Content to check
   * @returns {Object} Check result
   */
  checkPrimarySourceVerification(content) {
    // Look for claims that should be cited
    const claimsPattern = /\b(according to|as shown in|demonstrated by|research indicates|studies show|data shows|results show)\b/i;
    const hasClaims = claimsPattern.test(content);

    // Check if claims have citations
    const citationPattern = /\([A-Z][a-z]+, \d{4}\)|\[[0-9]+\]/g;
    const hasCitations = citationPattern.test(content);

    return {
      passed: hasClaims ? hasCitations : true,
      message: hasClaims && !hasCitations
        ? 'Content contains claims that should be supported by primary sources'
        : 'Primary source verification requirement satisfied'
    };
  }

  /**
   * Check academic clarity and precision
   * @param {string} content - Content to check
   * @returns {Object} Check result
   */
  checkAcademicClarity(content) {
    // Check for precise terminology and technical language
    const technicalTerms = content.match(/\b(ROS|Gazebo|Unity|NVIDIA|Isaac|VLA|AI|ML|SLAM|PID|TF|DDS|QoS|Docker|Kubernetes|API|SDK|JSON|XML|YAML|HTTP|TCP|IP|UART|I2C|SPI|PWM|ADC|GPIO|CPU|GPU|RAM|ROM|EEPROM|FPGA|ASIC|SoC|RTOS|OS|Linux|Ubuntu|Windows|macOS|Git|GitHub|CI|CD|IDE|SDK|IDE|CLI|GUI|IoT|IoE|IIoT|5G|4G|WiFi|Bluetooth|Zigbee|LoRaWAN|CAN|LIN|FlexRay|Ethernet|USB|UART|SPI|I2C|GPIO|PWM|ADC|DAC|UART|I2C|SPI|GPIO|PWM|ADC|DAC|CPU|GPU|RAM|ROM|EEPROM|FPGA|ASIC|SoC|RTOS|OS|Linux|Ubuntu|Windows|macOS|Git|GitHub|CI|CD|IDE|SDK|IDE|CLI|GUI|IoT|IoE|IIoT|5G|4G|WiFi|Bluetooth|Zigbee|LoRaWAN|CAN|LIN|FlexRay|Ethernet|USB)\b/gi);

    // Check for ambiguous language
    const ambiguousPattern = /\b(maybe|perhaps|sort of|kind of|a bit|quite|very|really|pretty|possibly|probably|likely|maybe|might|could|should|would|can|will|shall)\b/gi;
    const ambiguousMatches = content.match(ambiguousPattern) || [];

    return {
      passed: technicalTerms && ambiguousMatches.length < 5, // Allow some ambiguous language but not excessive
      message: technicalTerms && ambiguousMatches.length < 5
        ? 'Content demonstrates academic clarity and precision'
        : `Content contains ${ambiguousMatches.length} potentially ambiguous terms that may need clarification`
    };
  }

  /**
   * Check reproducibility and traceability
   * @param {string} content - Content to check
   * @returns {Object} Check result
   */
  checkReproducibility(content) {
    // Look for instructions or procedures that should be reproducible
    const procedurePattern = /\b(step|procedure|process|method|technique|approach|algorithm|implementation|configuration|setup|installation|deployment|execution|running|executing|performing|carrying out|following|applying)\b/i;
    const hasProcedures = procedurePattern.test(content);

    // Check for required details for reproduction
    const detailPatterns = [
      /code samples|example|sample|template|snippet|function|method|class|module|package/,
      /configuration|settings|parameters|arguments|variables/,
      /version|requirements|dependencies|prerequisites|setup/,
      /output|result|expected|verify|confirm|test|validate/
    ];

    const hasDetails = detailPatterns.some(pattern => pattern.test(content));

    return {
      passed: !hasProcedures || hasDetails,
      message: hasProcedures && !hasDetails
        ? 'Content contains procedures that may need more details for reproducibility'
        : 'Content provides sufficient detail for reproducibility'
    };
  }

  /**
   * Check scholarly rigor
   * @param {string} content - Content to check
   * @returns {Object} Check result
   */
  checkScholarlyRigor(content) {
    // Check for academic tone and references to peer-reviewed sources
    const academicIndicators = [
      /study|research|experiment|analysis|investigation|examination|review|survey/,
      /data|results|findings|conclusions|evidence|proof|demonstration/,
      /compared to|in contrast|similar to|different from|relative to/,
      /statistical|quantitative|qualitative|empirical|methodology|methodological/
    ];

    const hasAcademicIndicators = academicIndicators.some(pattern => pattern.test(content));

    return {
      passed: hasAcademicIndicators,
      message: hasAcademicIndicators
        ? 'Content demonstrates scholarly rigor'
        : 'Content may need more academic rigor with studies, data, or methodological approaches'
    };
  }

  /**
   * Check source verification
   * @param {string} content - Content to check
   * @returns {Object} Check result
   */
  checkSourceVerification(content) {
    // Check for claims without citations
    const claimPatterns = [
      /research shows|studies indicate|data shows|results demonstrate|findings suggest/,
      /it is known|it is established|it is proven|it is demonstrated/,
      /according to research|based on studies|from data/
    ];

    const hasClaims = claimPatterns.some(pattern => pattern.test(content));
    const hasCitations = /\([A-Z][a-z]+, \d{4}\)|\[[0-9]+\]/g.test(content);

    return {
      passed: !hasClaims || hasCitations,
      message: hasClaims && !hasCitations
        ? 'Content contains claims that require source verification'
        : 'All claims appear to be properly sourced'
    };
  }

  /**
   * Check citation requirements (APA format)
   * @param {string} content - Content to check
   * @returns {Object} Check result
   */
  checkCitationRequirements(content) {
    // Check for APA-style citations (Author, Year) or [Number]
    const apaPattern = /\([A-Z][a-z]+, \d{4}\)|\([A-Z][a-z]+ et al\., \d{4}\)/g;
    const numberedPattern = /\[[0-9]+\]/g;

    const apaMatches = content.match(apaPattern) || [];
    const numberedMatches = content.match(numberedPattern) || [];

    const hasValidCitations = apaMatches.length > 0 || numberedMatches.length > 0;

    return {
      passed: hasValidCitations,
      message: hasValidCitations
        ? 'Content follows APA citation requirements'
        : 'Content needs APA-style citations: (Author, Year) or [Number] format'
    };
  }

  /**
   * Run all constitution checks on content
   * @param {string} content - Content to check
   * @returns {Array} Array of check results
   */
  runConstitutionChecks(content) {
    const results = [];

    for (const [key, requirement] of Object.entries(this.constitutionRequirements)) {
      const result = requirement.checkFunction.call(this, content);
      results.push({
        id: key,
        name: requirement.name,
        description: requirement.description,
        passed: result.passed,
        message: result.message
      });
    }

    return results;
  }

  /**
   * Validate a markdown file against constitution requirements
   * @param {string} filePath - Path to the file to validate
   * @returns {Object} Validation results
   */
  validateFile(filePath) {
    if (!fs.existsSync(filePath)) {
      throw new Error(`File does not exist: ${filePath}`);
    }

    const content = fs.readFileSync(filePath, 'utf8');
    const checks = this.runConstitutionChecks(content);

    const passedCount = checks.filter(c => c.passed).length;
    const totalCount = checks.length;
    const isValid = passedCount === totalCount;

    return {
      filePath,
      isValid,
      passedCount,
      totalCount,
      complianceRate: Math.round((passedCount / totalCount) * 100),
      checks
    };
  }

  /**
   * Validate all markdown files in a directory
   * @param {string} dirPath - Directory to validate
   * @returns {Object} Overall validation results
   */
  validateDirectory(dirPath) {
    if (!fs.existsSync(dirPath)) {
      throw new Error(`Directory does not exist: ${dirPath}`);
    }

    const results = [];
    const files = this.getMarkdownFiles(dirPath);

    for (const file of files) {
      try {
        const result = this.validateFile(file);
        results.push(result);
      } catch (error) {
        console.error(`Error validating ${file}: ${error.message}`);
      }
    }

    const totalFiles = results.length;
    const compliantFiles = results.filter(r => r.isValid).length;
    const overallComplianceRate = totalFiles > 0 ? Math.round((compliantFiles / totalFiles) * 100) : 0;

    return {
      directory: dirPath,
      totalFiles,
      compliantFiles,
      nonCompliantFiles: totalFiles - compliantFiles,
      overallComplianceRate,
      fileResults: results
    };
  }

  /**
   * Get all markdown files in a directory recursively
   * @param {string} dir - Directory to scan
   * @returns {string[]} Array of markdown file paths
   */
  getMarkdownFiles(dir) {
    if (!fs.existsSync(dir)) return [];

    const files = [];
    const items = fs.readdirSync(dir);

    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        files.push(...this.getMarkdownFiles(fullPath));
      } else if (item.endsWith('.md')) {
        files.push(fullPath);
      }
    }

    return files;
  }

  /**
   * Generate a summary report of constitution compliance
   * @param {Array} results - Array of validation results
   * @returns {string} Summary report
   */
  generateSummaryReport(results) {
    if (!Array.isArray(results) || results.length === 0) {
      return 'No validation results to report.';
    }

    let report = 'Constitution Compliance Report\n';
    report += '=============================\n\n';

    if (results.length === 1 && results[0].checks) {
      // Single file validation
      const result = results[0];
      report += `File: ${result.filePath}\n`;
      report += `Status: ${result.isValid ? '✅ COMPLIANT' : '❌ NON-COMPLIANT'} (${result.complianceRate}%)\n\n`;

      for (const check of result.checks) {
        report += `- ${check.name}: ${check.passed ? '✅' : '❌'} ${check.message}\n`;
      }
    } else {
      // Directory validation
      const totalFiles = results.length;
      const compliantFiles = results.filter(r => r.isValid).length;
      const complianceRate = Math.round((compliantFiles / totalFiles) * 100);

      report += `Directory: ${results[0].directory}\n`;
      report += `Files: ${compliantFiles}/${totalFiles} compliant (${complianceRate}%)\n\n`;

      for (const result of results) {
        report += `- ${result.filePath}: ${result.isValid ? '✅' : '❌'} (${result.complianceRate}%)\n`;
      }
    }

    return report;
  }
}

module.exports = ContentReview;