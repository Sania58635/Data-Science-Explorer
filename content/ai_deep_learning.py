"""
AI, Deep Learning & NLP Content Module
Advanced neural networks and modern AI techniques
"""

def get_content():
    return {
        'summary': """
        <div class="topic-summary">
            <h3>📋 What is AI, Deep Learning & NLP?</h3>
            <p>Deep Learning uses multi-layer neural networks to automatically learn patterns from data, powering modern AI applications. NLP (Natural Language Processing) enables computers to understand and generate human language, from chatbots to translation.</p>
            <p><strong>Used in:</strong> Image Recognition (medical imaging, self-driving cars), Chatbots (GPT, Claude), Translation, Voice Assistants, Content Generation, and Recommendation Systems.</p>
        </div>
        """,
        'topics': [
            {
                'id': 'neural-networks',
                'title': 'Neural Network Architectures',
                'description': 'Foundational deep learning models',
                'subtopics': [
                    {
                        'name': 'Feedforward Neural Networks',
                        'content': """
                        <h3>Deep Neural Networks (DNN)</h3>
                        <p><strong>Architecture:</strong> Input layer → Hidden layers → Output layer</p>
                        <p><strong>Components:</strong></p>
                        <ul>
                            <li><strong>Neurons:</strong> Weighted sum + activation function</li>
                            <li><strong>Layers:</strong> Dense (fully connected) layers</li>
                            <li><strong>Weights:</strong> Learned parameters through backpropagation</li>
                        </ul>
                        <p><strong>Activation Functions:</strong></p>
                        <ul>
                            <li><strong>ReLU:</strong> max(0, x) → most common, prevents vanishing gradients</li>
                            <li><strong>Sigmoid:</strong> 1/(1+e^-x) → binary classification output</li>
                            <li><strong>Tanh:</strong> Centered at zero, range [-1, 1]</li>
                            <li><strong>Softmax:</strong> Multi-class probabilities (output layer)</li>
                            <li><strong>Leaky ReLU:</strong> Allows small negative values</li>
                        </ul>
                        <div class="visual">
                            <img src="/static/images/neural_network_architecture.png" alt="Neural Network Architecture" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="visual">
                            <img src="/static/images/activation_functions.png" alt="Activation Functions" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <h4>Building a Neural Network</h4>
                            <pre><code>import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'Convolutional Neural Networks (CNN)',
                        'content': """
                        <h3>CNNs for Computer Vision</h3>
                        <p><strong>Key Operations:</strong></p>
                        <ul>
                            <li><strong>Convolution Layer:</strong> Apply filters to detect features (edges, textures)</li>
                            <li><strong>Pooling Layer:</strong> Downsample spatial dimensions (max/average pooling)</li>
                            <li><strong>Flatten Layer:</strong> Convert 2D to 1D for dense layers</li>
                        </ul>
                        <p><strong>Architecture Pattern:</strong> Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → Dense</p>
                        <p><strong>Advantages:</strong></p>
                        <ul>
                            <li>Parameter sharing reduces computation</li>
                            <li>Translation invariance</li>
                            <li>Hierarchical feature learning</li>
                        </ul>
                        <p><strong>Applications:</strong> Image classification, object detection, segmentation, face recognition</p>
                        """
                    },
                    {
                        'name': 'Recurrent Neural Networks (RNN)',
                        'content': """
                        <h3>RNNs for Sequential Data</h3>
                        <p><strong>Concept:</strong> Networks with loops, maintaining hidden state across time steps</p>
                        <p><strong>Challenges:</strong></p>
                        <ul>
                            <li><strong>Vanishing Gradients:</strong> Long-term dependencies difficult to learn</li>
                            <li><strong>Exploding Gradients:</strong> Unstable training</li>
                        </ul>
                        <p><strong>Applications:</strong> Time series forecasting, text generation, speech recognition</p>
                        """
                    },
                    {
                        'name': 'LSTM (Long Short-Term Memory)',
                        'content': """
                        <h3>LSTM Networks</h3>
                        <p><strong>Innovation:</strong> Gating mechanisms control information flow</p>
                        <p><strong>Gates:</strong></p>
                        <ul>
                            <li><strong>Forget Gate:</strong> Decide what to discard from cell state</li>
                            <li><strong>Input Gate:</strong> Update cell state with new information</li>
                            <li><strong>Output Gate:</strong> Control what to output from cell state</li>
                        </ul>
                        <p><strong>Advantages:</strong> Captures long-term dependencies, mitigates vanishing gradients</p>
                        <p><strong>Use Cases:</strong> Machine translation, sentiment analysis, music generation</p>
                        <p><strong>Variant:</strong> GRU (Gated Recurrent Unit) → simpler, faster, similar performance</p>
                        """
                    },
                    {
                        'name': 'Transformers',
                        'content': """
                        <h3>Transformer Architecture</h3>
                        <p><strong>Revolutionary Concept:</strong> Attention mechanism replaces recurrence</p>
                        <p><strong>Self-Attention:</strong> Compute relevance between all input positions simultaneously</p>
                        <ul>
                            <li>Query, Key, Value matrices</li>
                            <li>Attention(Q,K,V) = softmax(QK^T/√d)V</li>
                            <li>Parallel processing (no sequential bottleneck)</li>
                        </ul>
                        <p><strong>Multi-Head Attention:</strong> Multiple attention mechanisms in parallel</p>
                        <p><strong>Positional Encoding:</strong> Inject sequence position information</p>
                        <p><strong>Architecture:</strong> Encoder-decoder with attention layers</p>
                        <p><strong>Impact:</strong> Foundation of modern NLP (BERT, GPT, T5)</p>
                        """
                    }
                ]
            },
            {
                'id': 'transfer-learning',
                'title': 'Transfer Learning',
                'description': 'Leveraging pretrained models',
                'subtopics': [
                    {
                        'name': 'Computer Vision Transfer Learning',
                        'content': """
                        <h3>Pretrained CNN Models</h3>
                        <p><strong>Popular Architectures:</strong></p>
                        <ul>
                            <li><strong>ResNet:</strong> Residual connections enable very deep networks (50-152 layers)</li>
                            <li><strong>VGG:</strong> Simple, deep architecture with small filters</li>
                            <li><strong>Inception:</strong> Multi-scale feature extraction</li>
                            <li><strong>EfficientNet:</strong> Optimized scaling for efficiency</li>
                            <li><strong>Vision Transformer (ViT):</strong> Transformer for images</li>
                        </ul>
                        <p><strong>Fine-Tuning Strategy:</strong></p>
                        <ol>
                            <li>Load pretrained weights (ImageNet)</li>
                            <li>Freeze early layers (general features)</li>
                            <li>Replace/retrain final layers (task-specific)</li>
                            <li>Optionally unfreeze and fine-tune entire network</li>
                        </ol>
                        <div class="visual">
                            <img src="/static/images/training_curves.png" alt="Training Progress" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        <div class="example">
                            <pre><code>from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze base

model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])</code></pre>
                        </div>
                        """
                    },
                    {
                        'name': 'NLP Transfer Learning',
                        'content': """
                        <h3>Pretrained Language Models</h3>
                        <p><strong>BERT (Bidirectional Encoder Representations):</strong></p>
                        <ul>
                            <li>Pretrained on masked language modeling</li>
                            <li>Bidirectional context understanding</li>
                            <li>Fine-tune for classification, NER, QA</li>
                        </ul>
                        <p><strong>GPT (Generative Pretrained Transformer):</strong></p>
                        <ul>
                            <li>Autoregressive language modeling</li>
                            <li>Excellent for text generation</li>
                            <li>Few-shot and zero-shot learning capabilities</li>
                        </ul>
                        <p><strong>T5 (Text-to-Text Transfer Transformer):</strong> Unified text-to-text framework</p>
                        <p><strong>RoBERTa:</strong> Optimized BERT training (more data, longer training)</p>
                        <p><strong>ELECTRA:</strong> Efficient pretraining via discriminative task</p>
                        """
                    }
                ]
            },
            {
                'id': 'nlp',
                'title': 'Natural Language Processing',
                'description': 'Understanding and generating human language',
                'subtopics': [
                    {
                        'name': 'Text Preprocessing',
                        'content': """
                        <h3>NLP Preprocessing Pipeline</h3>
                        <p><strong>Tokenization:</strong> Split text into words/subwords</p>
                        <ul>
                            <li>Word tokenization</li>
                            <li>Sentence tokenization</li>
                            <li>Subword tokenization (BPE, WordPiece)</li>
                        </ul>
                        <p><strong>Normalization:</strong></p>
                        <ul>
                            <li>Lowercasing</li>
                            <li>Removing punctuation</li>
                            <li>Stemming (reduce to root form)</li>
                            <li>Lemmatization (reduce to dictionary form)</li>
                        </ul>
                        <p><strong>Stopword Removal:</strong> Filter common words (the, is, at)</p>
                        <p><strong>Special Handling:</strong> Emojis, URLs, mentions, hashtags</p>
                        """
                    },
                    {
                        'name': 'Word Embeddings',
                        'content': """
                        <h3>Distributed Representations</h3>
                        <p><strong>Word2Vec:</strong> Neural network predicts context words</p>
                        <ul>
                            <li><strong>CBOW:</strong> Predict word from context</li>
                            <li><strong>Skip-gram:</strong> Predict context from word</li>
                            <li>Captures semantic similarity (king - man + woman ≈ queen)</li>
                        </ul>
                        <p><strong>GloVe:</strong> Global vectors from co-occurrence statistics</p>
                        <p><strong>FastText:</strong> Subword embeddings (handles rare words)</p>
                        <p><strong>Contextual Embeddings:</strong></p>
                        <ul>
                            <li><strong>ELMo:</strong> Context-dependent representations</li>
                            <li><strong>BERT embeddings:</strong> Bidirectional context</li>
                            <li>Same word, different meanings based on context</li>
                        </ul>
                        """
                    },
                    {
                        'name': 'Sentiment Analysis',
                        'content': """
                        <h3>Sentiment Classification</h3>
                        <p><strong>Task:</strong> Determine emotional tone (positive/negative/neutral)</p>
                        <p><strong>Approaches:</strong></p>
                        <ul>
                            <li><strong>Lexicon-Based:</strong> Sentiment dictionaries (VADER, TextBlob)</li>
                            <li><strong>Machine Learning:</strong> Train classifiers on labeled data</li>
                            <li><strong>Deep Learning:</strong> LSTM, CNN, Transformer models</li>
                        </ul>
                        <p><strong>Challenges:</strong></p>
                        <ul>
                            <li>Sarcasm and irony</li>
                            <li>Context-dependent sentiment</li>
                            <li>Domain-specific language</li>
                            <li>Aspect-based sentiment (multiple targets)</li>
                        </ul>
                        <p><strong>Applications:</strong> Brand monitoring, customer feedback, market research</p>
                        """
                    },
                    {
                        'name': 'Named Entity Recognition',
                        'content': """
                        <h3>NER and Information Extraction</h3>
                        <p><strong>Task:</strong> Identify and classify named entities (person, organization, location)</p>
                        <p><strong>Methods:</strong></p>
                        <ul>
                            <li><strong>Rule-Based:</strong> Patterns and gazetteers</li>
                            <li><strong>CRF:</strong> Conditional Random Fields for sequence labeling</li>
                            <li><strong>BiLSTM-CRF:</strong> Neural sequence labeling</li>
                            <li><strong>Transformer:</strong> BERT fine-tuning for NER</li>
                        </ul>
                        <p><strong>Extensions:</strong> Relation extraction, event detection, coreference resolution</p>
                        """
                    }
                ]
            },
            {
                'id': 'computer-vision',
                'title': 'Computer Vision',
                'description': 'Visual perception and understanding',
                'subtopics': [
                    {
                        'name': 'Image Classification',
                        'content': """
                        <h3>Image Recognition Tasks</h3>
                        <p><strong>Task:</strong> Assign label to entire image</p>
                        <p><strong>Architecture:</strong> CNN → Global pooling → Dense layers → Softmax</p>
                        <p><strong>Techniques:</strong></p>
                        <ul>
                            <li><strong>Data Augmentation:</strong> Rotation, flip, crop, color jitter</li>
                            <li><strong>Transfer Learning:</strong> Fine-tune pretrained models</li>
                            <li><strong>Ensemble Methods:</strong> Combine multiple models</li>
                        </ul>
                        <p><strong>Applications:</strong> Medical diagnosis, quality control, wildlife monitoring</p>
                        <div class="visual">
                            <img src="/static/images/confusion_matrix.png" alt="Confusion Matrix" style="max-width: 100%; border-radius: 12px; margin: 20px 0;">
                        </div>
                        """
                    },
                    {
                        'name': 'Object Detection',
                        'content': """
                        <h3>Detecting and Localizing Objects</h3>
                        <p><strong>Task:</strong> Find bounding boxes and classify multiple objects</p>
                        <p><strong>Architectures:</strong></p>
                        <ul>
                            <li><strong>R-CNN Family:</strong> Region proposals + classification (Faster R-CNN)</li>
                            <li><strong>YOLO:</strong> Single-shot detection (real-time performance)</li>
                            <li><strong>SSD:</strong> Multi-scale feature maps</li>
                            <li><strong>EfficientDet:</strong> Efficient architecture scaling</li>
                        </ul>
                        <p><strong>Evaluation:</strong> Intersection over Union (IoU), mAP (mean Average Precision)</p>
                        """
                    },
                    {
                        'name': 'Semantic Segmentation',
                        'content': """
                        <h3>Pixel-Level Classification</h3>
                        <p><strong>Task:</strong> Classify every pixel in image</p>
                        <p><strong>Architectures:</strong></p>
                        <ul>
                            <li><strong>FCN:</strong> Fully Convolutional Networks</li>
                            <li><strong>U-Net:</strong> Encoder-decoder with skip connections (medical imaging)</li>
                            <li><strong>DeepLab:</strong> Atrous convolution and pyramid pooling</li>
                            <li><strong>Mask R-CNN:</strong> Instance segmentation (separate object instances)</li>
                        </ul>
                        <p><strong>Applications:</strong> Autonomous driving, medical imaging, satellite imagery</p>
                        """
                    }
                ]
            },
            {
                'id': 'generative-ai',
                'title': 'Generative AI',
                'description': 'Creating new content with AI',
                'subtopics': [
                    {
                        'name': 'Generative Adversarial Networks (GANs)',
                        'content': """
                        <h3>GANs Architecture</h3>
                        <p><strong>Components:</strong></p>
                        <ul>
                            <li><strong>Generator:</strong> Creates fake samples from noise</li>
                            <li><strong>Discriminator:</strong> Distinguishes real from fake</li>
                            <li><strong>Training:</strong> Adversarial game → generator improves</li>
                        </ul>
                        <p><strong>Variants:</strong></p>
                        <ul>
                            <li><strong>DCGAN:</strong> Deep Convolutional GAN</li>
                            <li><strong>StyleGAN:</strong> Style-based generator (photorealistic faces)</li>
                            <li><strong>CycleGAN:</strong> Unpaired image-to-image translation</li>
                            <li><strong>Pix2Pix:</strong> Paired image translation</li>
                        </ul>
                        <p><strong>Applications:</strong> Image synthesis, data augmentation, art generation</p>
                        """
                    },
                    {
                        'name': 'Variational Autoencoders (VAE)',
                        'content': """
                        <h3>VAE Architecture</h3>
                        <p><strong>Components:</strong></p>
                        <ul>
                            <li><strong>Encoder:</strong> Maps input to latent distribution</li>
                            <li><strong>Latent Space:</strong> Probabilistic representation</li>
                            <li><strong>Decoder:</strong> Reconstructs from latent sample</li>
                        </ul>
                        <p><strong>Loss:</strong> Reconstruction loss + KL divergence (regularization)</p>
                        <p><strong>Applications:</strong> Anomaly detection, dimensionality reduction, generation</p>
                        """
                    },
                    {
                        'name': 'Diffusion Models',
                        'content': """
                        <h3>Denoising Diffusion Models</h3>
                        <p><strong>Concept:</strong> Gradually denoise random noise to generate samples</p>
                        <p><strong>Process:</strong></p>
                        <ul>
                            <li><strong>Forward:</strong> Add noise progressively</li>
                            <li><strong>Reverse:</strong> Learn to denoise step by step</li>
                        </ul>
                        <p><strong>Notable Models:</strong></p>
                        <ul>
                            <li><strong>DALL-E:</strong> Text-to-image generation</li>
                            <li><strong>Stable Diffusion:</strong> Latent diffusion for efficiency</li>
                            <li><strong>Midjourney:</strong> Artistic image generation</li>
                        </ul>
                        <p><strong>Advantages:</strong> High-quality samples, stable training, diverse outputs</p>
                        """
                    },
                    {
                        'name': 'Prompt Engineering',
                        'content': """
                        <h3>Optimizing LLM Interactions</h3>
                        <p><strong>Techniques:</strong></p>
                        <ul>
                            <li><strong>Zero-Shot:</strong> Task description only</li>
                            <li><strong>Few-Shot:</strong> Provide examples in prompt</li>
                            <li><strong>Chain-of-Thought:</strong> Request step-by-step reasoning</li>
                            <li><strong>Role-Playing:</strong> Define AI persona/expertise</li>
                        </ul>
                        <p><strong>Best Practices:</strong></p>
                        <ul>
                            <li>Be specific and clear</li>
                            <li>Provide context and constraints</li>
                            <li>Iterate and refine prompts</li>
                            <li>Use delimiters for structure</li>
                        </ul>
                        <p><strong>Applications:</strong> Code generation, content creation, data extraction, reasoning tasks</p>
                        """
                    }
                ]
            }
        ]
    }

