---
sidebar_position: 2
---

# Multimodal Embeddings and Representation

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the principles of multimodal embeddings for vision-language-action systems
2. Implement unified embedding spaces that connect visual, linguistic, and action modalities
3. Design cross-modal attention mechanisms for information fusion
4. Evaluate multimodal embedding quality and alignment
5. Optimize embedding architectures for real-time robotic applications

## Introduction to Multimodal Embeddings

Multimodal embeddings form the foundation of Vision-Language-Action (VLA) systems by creating unified representations that capture relationships across different modalities. Unlike unimodal embeddings that operate within a single domain (e.g., text embeddings for language or visual embeddings for images), multimodal embeddings enable cross-domain reasoning by mapping different types of information to a shared semantic space.

The challenge in multimodal embeddings lies in creating representations that preserve the unique characteristics of each modality while enabling meaningful comparisons and interactions between them. For VLA systems, this means connecting:

- **Visual Embeddings**: Representations of images, video, and sensor data
- **Linguistic Embeddings**: Representations of text, commands, and descriptions
- **Action Embeddings**: Representations of robot movements, trajectories, and behaviors

## Theoretical Foundations

### Embedding Spaces and Modalities

An embedding space is a continuous vector space where discrete inputs (words, images, actions) are mapped to dense vector representations. In multimodal systems, we have multiple embedding spaces that need to be aligned:

```
Visual Space:    R^D_v  ← Images, video, sensor data
Linguistic Space: R^D_l  ← Text, commands, descriptions
Action Space:     R^D_a  ← Motor commands, trajectories
Joint Space:      R^D_j  ← Unified multimodal representations
```

The goal is to learn mappings between these spaces such that semantically related concepts across modalities are close in the joint embedding space.

### Cross-Modal Alignment

Cross-modal alignment ensures that related concepts from different modalities are mapped to nearby locations in the embedding space. For example:

- The image of a "red ball" should be close to the text "red ball"
- The action "pick up the ball" should be close to both the image and text representations
- The trajectory of reaching toward a ball should be semantically related to both visual and linguistic concepts

## Architectural Approaches

### Early Fusion vs. Late Fusion

Two primary approaches exist for combining multimodal information:

#### Early Fusion
In early fusion, different modalities are combined at an early stage of processing:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyFusionMultimodalEncoder(nn.Module):
    def __init__(self, visual_dim, text_dim, action_dim, joint_dim):
        super().__init__()

        # Individual modality encoders
        self.visual_encoder = nn.Linear(visual_dim, joint_dim)
        self.text_encoder = nn.Linear(text_dim, joint_dim)
        self.action_encoder = nn.Linear(action_dim, joint_dim)

        # Fusion layer that combines all modalities
        self.fusion_layer = nn.Linear(joint_dim * 3, joint_dim)
        self.norm = nn.LayerNorm(joint_dim)

    def forward(self, visual_input, text_input, action_input):
        # Encode each modality to joint space
        visual_emb = self.visual_encoder(visual_input)
        text_emb = self.text_encoder(text_input)
        action_emb = self.action_encoder(action_input)

        # Concatenate and fuse
        combined = torch.cat([visual_emb, text_emb, action_emb], dim=-1)
        fused = self.fusion_layer(combined)

        return self.norm(fused)
```

#### Late Fusion
In late fusion, modalities are processed separately and combined at a later stage:

```python
class LateFusionMultimodalEncoder(nn.Module):
    def __init__(self, visual_dim, text_dim, action_dim, joint_dim):
        super().__init__()

        # Individual encoders
        self.visual_encoder = nn.Linear(visual_dim, joint_dim)
        self.text_encoder = nn.Linear(text_dim, joint_dim)
        self.action_encoder = nn.Linear(action_dim, joint_dim)

        # Cross-attention for fusion
        self.cross_attention = nn.MultiheadAttention(joint_dim, num_heads=8)
        self.fusion_transformer = nn.TransformerEncoderLayer(
            d_model=joint_dim, nhead=8, dim_feedforward=joint_dim*2
        )

    def forward(self, visual_input, text_input, action_input):
        # Encode each modality separately
        visual_emb = self.visual_encoder(visual_input)  # [B, seq_len_v, D]
        text_emb = self.text_encoder(text_input)       # [B, seq_len_t, D]
        action_emb = self.action_encoder(action_input) # [B, seq_len_a, D]

        # Concatenate sequences
        combined_seq = torch.cat([visual_emb, text_emb, action_emb], dim=1)

        # Apply cross-modal attention
        fused_output = self.fusion_transformer(combined_seq)

        return fused_output
```

### Cross-Modal Attention Mechanisms

Cross-modal attention allows information from one modality to influence the representation of another:

```python
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query, key, value projections for cross-attention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query_modality, key_modality, value_modality):
        B, N, C = query_modality.shape

        # Project to query, key, value
        q = self.q_proj(query_modality).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_modality).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value_modality).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, N, C)

        return self.out_proj(output)

class MultimodalFusionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.visual_to_text = CrossModalAttention(dim)
        self.text_to_visual = CrossModalAttention(dim)
        self.action_to_joint = CrossModalAttention(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, visual_features, text_features, action_features):
        # Visual features influence text representation
        text_updated = self.norm1(text_features + self.visual_to_text(
            text_features, visual_features, visual_features
        ))

        # Text features influence visual representation
        visual_updated = self.norm2(visual_features + self.text_to_visual(
            visual_features, text_features, text_features
        ))

        # Action features are updated based on joint representation
        joint_repr = torch.cat([visual_updated, text_updated], dim=-1)
        action_updated = self.norm3(action_features + self.action_to_joint(
            action_features, joint_repr, joint_repr
        ))

        return visual_updated, text_updated, action_updated
```

## Vision-Language Embeddings

### CLIP-Inspired Architecture

CLIP (Contrastive Language-Image Pre-training) provides a foundation for vision-language alignment:

```python
class VisionLanguageEncoder(nn.Module):
    def __init__(self, vision_model, text_model, embed_dim):
        super().__init__()

        self.vision_encoder = vision_model
        self.text_encoder = text_model
        self.visual_projection = nn.Linear(vision_model.dim, embed_dim)
        self.text_projection = nn.Linear(text_model.dim, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, images, texts):
        # Encode visual and text features
        visual_features = self.vision_encoder(images)  # [B, D_v]
        text_features = self.text_encoder(texts)      # [B, D_t]

        # Project to common embedding space
        visual_embeds = self.visual_projection(visual_features)  # [B, D]
        text_embeds = self.text_projection(text_features)      # [B, D]

        # Normalize embeddings
        visual_embeds = F.normalize(visual_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Compute similarity matrix
        logits_per_image = self.logit_scale * visual_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return {
            'visual_embeds': visual_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
```

### Vision-Language-Action Extension

Extending the vision-language framework to include action embeddings:

```python
class VisionLanguageActionEncoder(nn.Module):
    def __init__(self, vision_dim, text_dim, action_dim, embed_dim):
        super().__init__()

        # Individual encoders
        self.vision_encoder = nn.Linear(vision_dim, embed_dim)
        self.text_encoder = nn.Linear(text_dim, embed_dim)
        self.action_encoder = nn.Linear(action_dim, embed_dim)

        # Projection layers for contrastive learning
        self.vision_proj = nn.Linear(embed_dim, embed_dim)
        self.text_proj = nn.Linear(embed_dim, embed_dim)
        self.action_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_vision(self, images):
        features = self.vision_encoder(images)
        return F.normalize(self.vision_proj(features), dim=-1)

    def encode_text(self, texts):
        features = self.text_encoder(texts)
        return F.normalize(self.text_proj(features), dim=-1)

    def encode_action(self, actions):
        features = self.action_encoder(actions)
        return F.normalize(self.action_proj(features), dim=-1)

    def forward(self, images, texts, actions):
        # Encode all modalities
        image_embeds = self.encode_vision(images)
        text_embeds = self.encode_text(texts)
        action_embeds = self.encode_action(actions)

        # Compute similarity matrices for contrastive learning
        # Image-Text similarities
        logits_i2t = self.logit_scale * image_embeds @ text_embeds.t()
        logits_t2i = logits_i2t.t()

        # Image-Action similarities
        logits_i2a = self.logit_scale * image_embeds @ action_embeds.t()
        logits_a2i = logits_i2a.t()

        # Text-Action similarities
        logits_t2a = self.logit_scale * text_embeds @ action_embeds.t()
        logits_a2t = logits_t2a.t()

        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'action_embeds': action_embeds,
            'logits_i2t': logits_i2t,
            'logits_t2i': logits_t2i,
            'logits_i2a': logits_i2a,
            'logits_a2i': logits_a2i,
            'logits_t2a': logits_t2a,
            'logits_a2t': logits_a2t
        }
```

## Action Embedding Representations

### Continuous Action Spaces

For robotic systems, actions can be represented as continuous motor commands:

```python
class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, embed_dim):
        super().__init__()

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, actions):
        return self.action_encoder(actions)

class TrajectoryEncoder(nn.Module):
    def __init__(self, action_dim, hidden_dim, embed_dim, max_length=100):
        super().__init__()

        self.action_dim = action_dim
        self.max_length = max_length

        # Embed each action in the trajectory
        self.action_embedding = nn.Linear(action_dim, hidden_dim)

        # Process sequence with transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 2,
                batch_first=True
            ),
            num_layers=3
        )

        # Project to final embedding space
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, trajectories):
        # trajectories: [B, T, action_dim]
        B, T, _ = trajectories.shape

        # Embed each action
        action_embeds = self.action_embedding(trajectories)  # [B, T, hidden_dim]

        # Apply transformer to model temporal dependencies
        traj_embeds = self.transformer(action_embeds)  # [B, T, hidden_dim]

        # Aggregate over time dimension (e.g., mean pooling)
        final_embed = traj_embeds.mean(dim=1)  # [B, hidden_dim]

        # Project to embedding space
        return self.projection(final_embed)
```

### Discrete Action Spaces

For systems with discrete action spaces:

```python
class DiscreteActionEncoder(nn.Module):
    def __init__(self, num_actions, embed_dim):
        super().__init__()
        self.action_embedding = nn.Embedding(num_actions, embed_dim)

    def forward(self, action_indices):
        return self.action_embedding(action_indices)

class ActionSequenceEncoder(nn.Module):
    def __init__(self, num_actions, embed_dim, hidden_dim):
        super().__init__()

        self.action_embedding = nn.Embedding(num_actions, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, embed_dim)

    def forward(self, action_sequences):
        # action_sequences: [B, T] where T is sequence length
        embedded = self.action_embedding(action_sequences)  # [B, T, embed_dim]
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use final hidden state as sequence embedding
        return self.projection(hidden[-1])  # [B, embed_dim]
```

## Training Strategies

### Contrastive Learning

Contrastive learning is a key technique for training multimodal embeddings:

```python
def contrastive_loss(similarities, labels=None):
    """
    Compute contrastive loss for multimodal alignment
    similarities: similarity matrix between modalities
    """
    # For positive pairs (diagonal), we want high similarity
    # For negative pairs (off-diagonal), we want low similarity

    batch_size = similarities.size(0)

    # Create labels for contrastive learning
    if labels is None:
        labels = torch.arange(batch_size, device=similarities.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(similarities, labels)

    return loss

class MultimodalTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train_step(self, images, texts, actions):
        self.optimizer.zero_grad()

        outputs = self.model(images, texts, actions)

        # Compute contrastive losses for all modality pairs
        loss_i2t = contrastive_loss(outputs['logits_i2t'])
        loss_t2i = contrastive_loss(outputs['logits_t2i'])
        loss_i2a = contrastive_loss(outputs['logits_i2a'])
        loss_a2i = contrastive_loss(outputs['logits_a2i'])
        loss_t2a = contrastive_loss(outputs['logits_t2a'])
        loss_a2t = contrastive_loss(outputs['logits_a2t'])

        # Total loss
        total_loss = (loss_i2t + loss_t2i + loss_i2a +
                     loss_a2i + loss_t2a + loss_a2t) / 6

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
```

### Triplet Loss

Triplet loss can be used to ensure that related samples are closer than unrelated ones:

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Compute triplet loss: d(anchor, positive) + margin < d(anchor, negative)
    """
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)

    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

class TripletMultimodalTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train_step(self, anchors, positives, negatives):
        """
        anchors: [images, texts, or actions]
        positives: corresponding modality samples
        negatives: non-corresponding modality samples
        """
        self.optimizer.zero_grad()

        anchor_embeds = self.model.encode_vision(anchors)  # or text/action encoder
        pos_embeds = self.model.encode_text(positives)    # or appropriate encoder
        neg_embeds = self.model.encode_action(negatives)  # or appropriate encoder

        loss = triplet_loss(anchor_embeds, pos_embeds, neg_embeds)

        loss.backward()
        self.optimizer.step()

        return loss.item()
```

## Evaluation Metrics

### Cross-Modal Retrieval

Cross-modal retrieval evaluates how well embeddings from one modality can retrieve relevant samples from another:

```python
def evaluate_retrieval(embeddings1, embeddings2, k=1):
    """
    Evaluate cross-modal retrieval performance
    embeddings1: embeddings from modality 1
    embeddings2: embeddings from modality 2
    k: number of top results to consider
    """
    # Compute similarity matrix
    similarities = embeddings1 @ embeddings2.t()  # [N, N]

    # For each sample, find the rank of its corresponding sample
    ranks = []
    for i in range(len(embeddings1)):
        # Get similarities for sample i
        sample_similarities = similarities[i]

        # Sort in descending order and get indices
        _, indices = torch.sort(sample_similarities, descending=True)

        # Find rank of corresponding sample (diagonal element)
        rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)

    # Compute metrics
    ranks = torch.tensor(ranks)
    r1 = (ranks <= 1).float().mean()
    r5 = (ranks <= 5).float().mean()
    r10 = (ranks <= 10).float().mean()
    medr = ranks.median().item()
    meanr = ranks.mean().item()

    return {
        'R@1': r1.item(),
        'R@5': r5.item(),
        'R@10': r10.item(),
        'MedR': medr,
        'MeanR': meanr
    }

def evaluate_vla_system(vla_model, test_dataset):
    """Evaluate complete VLA system"""
    all_image_embeds = []
    all_text_embeds = []
    all_action_embeds = []

    with torch.no_grad():
        for batch in test_dataset:
            images, texts, actions = batch
            outputs = vla_model(images, texts, actions)

            all_image_embeds.append(outputs['image_embeds'])
            all_text_embeds.append(outputs['text_embeds'])
            all_action_embeds.append(outputs['action_embeds'])

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_action_embeds = torch.cat(all_action_embeds, dim=0)

    # Evaluate all cross-modal retrieval tasks
    results = {}
    results['image_to_text'] = evaluate_retrieval(all_image_embeds, all_text_embeds)
    results['text_to_image'] = evaluate_retrieval(all_text_embeds, all_image_embeds)
    results['image_to_action'] = evaluate_retrieval(all_image_embeds, all_action_embeds)
    results['action_to_image'] = evaluate_retrieval(all_action_embeds, all_image_embeds)
    results['text_to_action'] = evaluate_retrieval(all_text_embeds, all_action_embeds)
    results['action_to_text'] = evaluate_retrieval(all_action_embeds, all_text_embeds)

    return results
```

## Real-Time Optimization

### Efficient Embedding Computation

For real-time robotic applications, efficient embedding computation is crucial:

```python
class EfficientMultimodalEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        # Lightweight encoders for real-time performance
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(768, embed_dim),  # Assuming BERT-like input
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(7, embed_dim),  # Assuming 7-DOF robot joint positions
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, visual_input, text_input, action_input):
        visual_embed = self.visual_encoder(visual_input)
        text_embed = self.text_encoder(text_input)
        action_embed = self.action_encoder(action_input)

        return visual_embed, text_embed, action_embed

# Quantization for even faster inference
def quantize_model(model):
    """Apply quantization to reduce model size and improve inference speed"""
    import torch.quantization as quant

    model.eval()

    # Specify layers to quantize
    quant_backend = 'qnnpack'  # Use QNNPACK for mobile/edge
    model.qconfig = quant.get_default_qconfig(quant_backend)

    # Fuse operations for better quantization
    quant.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

    return model
```

### Caching and Precomputation

For frequently accessed embeddings, caching can significantly improve performance:

```python
class CachedMultimodalSystem:
    def __init__(self, model, cache_size=1000):
        self.model = model
        self.cache_size = cache_size
        self.embedding_cache = {}
        self.access_times = {}

    def get_or_compute_embedding(self, input_tensor, modality_type):
        """Get embedding from cache or compute if not present"""
        # Create a hashable key from the input tensor
        tensor_hash = hash(input_tensor.mean().item())  # Simplified hashing

        cache_key = f"{modality_type}_{tensor_hash}"

        if cache_key in self.embedding_cache:
            # Update access time for LRU
            self.access_times[cache_key] = time.time()
            return self.embedding_cache[cache_key]

        # Compute embedding
        with torch.no_grad():
            if modality_type == 'visual':
                embedding = self.model.encode_vision(input_tensor)
            elif modality_type == 'text':
                embedding = self.model.encode_text(input_tensor)
            elif modality_type == 'action':
                embedding = self.model.encode_action(input_tensor)

        # Add to cache
        self.embedding_cache[cache_key] = embedding
        self.access_times[cache_key] = time.time()

        # Evict oldest entries if cache is full
        if len(self.embedding_cache) > self.cache_size:
            oldest_key = min(self.access_times.keys(),
                           key=lambda k: self.access_times[k])
            del self.embedding_cache[oldest_key]
            del self.access_times[oldest_key]

        return embedding
```

## Isaac Integration

### Isaac ROS for Multimodal Processing

Integrating multimodal embeddings with Isaac ROS systems:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch
import numpy as np

class IsaacMultimodalNode(Node):
    def __init__(self):
        super().__init__('isaac_multimodal_node')

        self.bridge = CvBridge()

        # Load pre-trained multimodal model
        self.model = self.load_multimodal_model()
        self.model.eval()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_rect_color', self.image_callback, 10
        )

        self.text_sub = self.create_subscription(
            String, 'natural_language_command', self.text_callback, 10
        )

        self.command_pub = self.create_publisher(Twist, 'robot_velocity', 10)

        # Store latest embeddings
        self.latest_visual_embed = None
        self.latest_text_embed = None

    def load_multimodal_model(self):
        """Load pre-trained multimodal model"""
        # This would load your trained VLA model
        model = VisionLanguageActionEncoder(
            vision_dim=512, text_dim=512, action_dim=7, embed_dim=512
        )

        # Load pre-trained weights
        # model.load_state_dict(torch.load('vla_model.pth'))

        return model

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Preprocess image
            image_tensor = self.preprocess_image(cv_image)

            # Compute visual embedding
            with torch.no_grad():
                self.latest_visual_embed = self.model.encode_vision(image_tensor)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def text_callback(self, msg):
        """Process incoming text command"""
        try:
            # Tokenize and encode text
            text_tensor = self.tokenize_text(msg.data)

            # Compute text embedding
            with torch.no_grad():
                self.latest_text_embed = self.model.encode_text(text_tensor)

            # If we have both visual and text embeddings, generate action
            if self.latest_visual_embed is not None and self.latest_text_embed is not None:
                self.generate_action()

        except Exception as e:
            self.get_logger().error(f'Error processing text: {e}')

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize, normalize, convert to tensor
        import cv2
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return image_tensor

    def tokenize_text(self, text):
        """Tokenize text for model input"""
        # This would use your text tokenizer
        # For now, return a dummy tensor
        return torch.randn(1, 768)  # Placeholder

    def generate_action(self):
        """Generate robot action based on visual and text embeddings"""
        # Compute similarity between visual and text embeddings
        similarity = F.cosine_similarity(
            self.latest_visual_embed, self.latest_text_embed
        )

        # Based on similarity and text content, generate appropriate action
        command = Twist()

        if similarity > 0.8:  # High similarity indicates relevant command
            # This is a simplified example - in practice, you'd have more sophisticated logic
            command.linear.x = 0.5  # Move forward
            command.angular.z = 0.0  # No rotation

        self.command_pub.publish(command)
```

## Summary

Multimodal embeddings form the backbone of Vision-Language-Action systems by creating unified representations that connect visual, linguistic, and action modalities. Through careful design of embedding architectures, training strategies, and optimization techniques, we can build systems that understand the relationships between what robots see, what humans say, and what robots do.

The next section will explore instruction following and task planning, which builds upon these embedding foundations to enable robots to interpret and execute natural language commands.

## References

[All sources will be cited in the References section at the end of the book, following APA format]