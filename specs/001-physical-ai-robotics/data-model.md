# Data Model: Physical AI & Humanoid Robotics Book

## Book Content Structure

### Book
- **title**: Physical AI & Humanoid Robotics: Embodied Intelligence, Simulation-to-Reality Workflows, and Modern AI-Robot Integration
- **word_count**: 5000-7000 (excluding references)
- **target_audience**: Graduate students, researchers, software engineers
- **prerequisites**: Programming, linear algebra, basic ML
- **modules**: [Module 1, Module 2, Module 3, Module 4]
- **capstone_project**: "The Autonomous Humanoid"
- **deliverables**: [manuscript, chapter_pdfs, code_samples, simulation_assets, reference_list, capstone_report]

### Module
- **module_id**: (e.g., PIAHR-M1)
- **title**: (e.g., "The Robotic Nervous System (ROS 2)")
- **description**: Module overview
- **key_concepts**: [list of key concepts]
- **skills_gained**: [list of skills]
- **weekly_alignment**: (e.g., "Weeks 1-3")
- **deliverables_labs**: [list of deliverables]
- **verification_rules**: [list of verification rules]
- **explicit_exclusions**: [list of exclusions]

### Source
- **source_id**: Unique identifier
- **title**: Source title
- **author**: Author(s)
- **publication_date**: Date published
- **type**: (e.g., "peer-reviewed", "documentation", "standard")
- **url**: Accessible URL
- **is_peer_reviewed**: Boolean (must be true for ≥50% of sources)
- **citation**: APA format citation
- **topic_areas**: [list of related topics]

### CapstoneProject
- **title**: "The Autonomous Humanoid"
- **functional_requirements**: [list of requirements]
- **system_boundaries**: [input/output/environment specifications]
- **success_criteria**: [measurable success metrics]
- **evaluation_rubric**: [grading criteria breakdown]

### HardwareSpecification
- **component_type**: (e.g., "workstation", "edge_device", "sensor")
- **minimum_spec**: Minimum required specifications
- **recommended_spec**: Recommended specifications
- **rationale**: Reasoning for specifications
- **category**: (e.g., "Digital Twin Workstation", "Jetson Edge AI Kit")

### Assessment
- **assessment_type**: (e.g., "ROS 2 Package Project", "Gazebo Simulation")
- **evaluation_criteria**: [list of criteria]
- **pass_criteria**: Specific requirements for passing
- **grading_breakdown**: How points are allocated

## Validation Rules
- Book.word_count must be between 5000 and 7000 words (excluding references)
- At least 50% of sources must have is_peer_reviewed = true
- All modules must have valid module_id format (PIAHR-M[1-4])
- All citations must follow APA format
- All factual claims must be linked to a source
- Writing clarity must achieve Flesch-Kincaid grade level of 10-12