# Smart Infrastructure Complaint Intelligence System

**An AI-powered municipal complaint triage platform using Multi-Task BERT, spaCy NER, and real-time analytics**

This Streamlit application provides intelligent automation for civic infrastructure complaint management. It combines deep learning with rule-based heuristics to classify, prioritize, and extract actionable insights from citizen complaints in real-time. Designed for municipal control rooms and infrastructure agencies, it transforms raw complaint text into analyst-ready intelligence with automated categorization, severity scoring, urgency assessment, and entity extraction.

---

## 🎯 Key Features

### **1. Multi-Task BERT Classification**
- **31 Infrastructure Categories**: Electrical, Water Crisis, Solid Waste, Road Infrastructure, Storm Water Drain, Traffic Engineer Cell, Health Dept, and 24+ more civic service domains
- **Three-Head Architecture**: Single model simultaneously predicts:
  - **Category** (31 classes): Issue type classification
  - **Severity** (3 levels): Low, Medium, High
  - **Urgency** (3 levels): Neutral, Concerned, Angry/Urgent
- **Keyword Enhancement**: Rule-based boosting for critical keywords (emergency, dangerous, urgent, etc.)
- **Confidence Scoring**: Probabilistic outputs with interpretable confidence metrics

### **2. Intelligent Entity Extraction**
- **spaCy NER Pipeline**: Extracts locations (LOC, GPE), organizations, and problem terms
- **Automatic Location Detection**: Identifies civic landmarks, ward names, and geographic references
- **Inline Annotation**: Visual `word{LABEL}` format for quick triage
- **GIS-Ready Output**: Extracted locations ready for mapping and routing systems

### **3. Rule-Based Safety Nets**
- **Severity Rules** (`utils/severity_rules.py`): Keyword-based escalation for life-threatening situations
  - High: emergency, dangerous, death, collapse, explosion, fire, etc.
  - Medium: broken, damaged, overflowing, not working, etc.
  - Low: minor, cosmetic, aesthetic issues
- **Urgency Rules** (`utils/urgency_rules.py`): Emotional and time-sensitive language detection
  - Urgent: immediate, critical, cannot wait, lives at stake, etc.
  - Concerned: worried, risky, deteriorating, repeated complaints, etc.

### **4. Real-Time Analytics Dashboard**
- **Time-Series Trends**: Complaint volume over time with date filtering
- **Category Distribution**: Bar charts showing top issue types
- **Severity/Urgency Matrix**: Heatmap visualization for priority quadrants
- **Geographic Hotspots**: Location-based complaint clustering
- **Word Clouds**: Visual representation of frequently mentioned terms
- **Data Export**: Download CSV datasets with timestamps and analysis results

### **5. Dual Theme Support**
- **Light & Dark Modes**: Optimized color schemes for both themes
- **High Contrast**: All text readable with proper color contrast ratios
- **Smooth Transitions**: 0.6s animated theme switching
- **Professional Design**: Modern card-based UI with gradient hero sections

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│  (app.py - UI, navigation, visualization, theme management) │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Analysis Pipeline                               │
│  (utils/analysis_pipeline.py - orchestration & validation)  │
└─┬───────────────┬──────────────┬───────────────┬───────────┘
  │               │              │               │
  │               │              │               │
  ▼               ▼              ▼               ▼
┌───────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐
│ BERT  │  │ Severity │  │ Urgency  │  │   spaCy NER    │
│ Head  │  │   Head   │  │   Head   │  │ (en_core_web)  │
│ (31)  │  │   (3)    │  │   (3)    │  │ LOC/GPE/ORG    │
└───┬───┘  └────┬─────┘  └────┬─────┘  └────────┬───────┘
    │           │             │                  │
    │           │             │                  │
    └───────────┴─────────────┴──────────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Rule Enhancement      │
         │  - Keyword matching    │
         │  - Confidence boosting │
         │  - Rationale generation│
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────────┐
         │   Storage & Analytics  │
         │  - CSV persistence     │
         │  - Time-series data    │
         │  - Visualization prep  │
         └────────────────────────┘
```

---

## 📦 Installation & Setup

### **Prerequisites**
- Python 3.8+ (tested on 3.11)
- Conda environment manager
- 4GB+ RAM (for model loading)
- Windows/Linux/macOS

### **1. Clone Repository**
```bash
git clone https://github.com/prashanth-31/Smart-Infrastructure-Complaint-Intelligence-System-Using-NLP-LLM-Techniques.git
cd Smart-Infrastructure-Complaint-Intelligence-System-Using-NLP-LLM-Techniques
```

### **2. Create Environment**
```powershell
# Create and activate conda environment
conda create -n infrastructure-ai python=3.11
conda activate infrastructure-ai

# Or use existing venv
# conda activate E:\Python\venv
```

### **3. Install Dependencies**
```powershell
pip install -r requirements.txt
```

**Key Dependencies:**
- `streamlit` - Web application framework
- `transformers` - Hugging Face model loading
- `torch` - PyTorch deep learning
- `spacy` - NER pipeline
- `pandas` - Data manipulation
- `plotly`, `altair` - Interactive visualizations
- `wordcloud` - Text visualization

### **4. Verify Model Files**

Ensure the following directory structure exists:

```
models/
├── multi_task_classifier.pt          # Multi-Task BERT checkpoint
├── tokenizer/                         # BERT tokenizer files
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── tokenizer.json
└── spacy_model/
    └── en_core_web_sm/
        └── en_core_web_sm-3.8.0/     # spaCy NER model
```

**Check Installation:**
```powershell
python -c "from config import MODEL_FILES; import sys; missing = [k for k, v in MODEL_FILES.items() if not v.exists()]; print('✅ All models present' if not missing else f'❌ Missing: {missing}'); sys.exit(0 if not missing else 1)"
```

---

## 🚀 Running the Application

### **Start Streamlit Server**
```powershell
streamlit run app.py
```

The app will launch at `http://localhost:8501`

### **First-Time Setup**
1. **Theme Selection**: Toggle dark mode in sidebar
2. **Submit Test Complaint**: Try example text to verify pipeline
3. **Check Dashboard**: Navigate to "Dashboard & Analytics" tab
4. **Clear Dataset** (optional): Start fresh with empty CSV

### **Example Complaint**
```
There is a dangerous electrical wire hanging loose near Gandhi Nagar school. 
Children are at risk and this needs immediate attention. The street light pole 
has collapsed after yesterday's storm.
```

**Expected Output:**
- **Category**: Electrical
- **Severity**: High (keywords: dangerous, collapsed)
- **Urgency**: Angry/Urgent (keywords: immediate, children at risk)
- **Location**: Gandhi Nagar
- **Entities**: Gandhi Nagar (LOC), school (ORG), electrical wire (PROBLEM)

---

## 📊 Model Architecture Details

### **Multi-Task BERT Classifier**
- **Base Model**: `bert-base-uncased` (110M parameters)
- **Architecture**:
  ```
  BERT Encoder (768-dim hidden states)
       │
       ├─► Category Head → Linear(768, 31) → Softmax → 31 categories
       ├─► Severity Head → Linear(768, 3)  → Softmax → Low/Medium/High
       └─► Urgency Head  → Linear(768, 3)  → Softmax → Neutral/Concerned/Urgent
  ```
- **Training Data**: `data/final_grievances_cleaned.csv` (126K+ civic complaints)
- **Checkpoint**: `models/multi_task_classifier.pt` (PyTorch state_dict)

### **Label Mappings**
**Categories (31):**
```python
["Advertisement", "BBMP Election Branch", "CORONA COVID19", "Call Center",
 "E khata / Khata services", "Education", "Electrical", "Estate",
 "Forest", "Health Dept", "Indira Canteen", "Information Technology",
 "Lakes", "Markets", "Optical Fiber Cables (OFC)", "Others",
 "Parks and Play grounds", "Plastic", "Projects Central", 
 "Property Tax services", "Revenue Department", "Road Infrastructure",
 "Road Maintenance(Engg)", "Sanitation", "Solid Waste (Garbage) Related",
 "Storm Water Drain(SWD)", "Town Planning", "Traffic Engineer Cell (TEC)",
 "Water Crisis", "Welfare Schemes", "veterinary"]
```

**Severity (3):** `Low`, `Medium`, `High`  
**Urgency (3):** `Neutral`, `Concerned`, `Angry/Urgent`

---

## 📁 Project Structure

```
Smart-Infrastructure-Complaint-Intelligence-System/
│
├── app.py                          # Main Streamlit application
├── config.py                       # Path configurations
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/                         # Model artifacts
│   ├── multi_task_classifier.pt    # Multi-Task BERT checkpoint
│   ├── tokenizer/                  # BERT tokenizer
│   ├── spacy_model/                # spaCy NER model
│   ├── pipeline_loader.py          # Model loading logic
│   └── __init__.py
│
├── utils/                          # Core utilities
│   ├── analysis_pipeline.py        # Main orchestration & inference
│   ├── severity_rules.py           # Severity keyword rules
│   ├── urgency_rules.py            # Urgency keyword rules
│   ├── preprocessing.py            # Text normalization
│   ├── validation.py               # Output validation
│   ├── data_store.py               # CSV persistence
│   ├── visualization.py            # Chart generation
│   └── severity_features.py        # Legacy feature extraction
│
├── data/                           # Datasets
│   ├── complaints_sample.csv       # Live complaint log
│   ├── final_grievances_cleaned.csv# Training data (126K rows)
│   └── 2025-complaints.csv         # Test data
│
├── assets/                         # Frontend assets
│   └── css/
│       └── app.css                 # Custom styling
│
├── training/                       # Training scripts
│   └── config.py                   # Training configuration
│
└── tests/                          # Unit tests
    ├── test_model_loading.py
    ├── test_validation.py
    └── test_thread_safety.py
```

---

## 🔧 Configuration

### **File Paths** (`config.py`)
```python
MODEL_FILES = {
    "multi_task_classifier": MODEL_DIR / "multi_task_classifier.pt",
    "tokenizer": MODEL_DIR / "tokenizer",
    "spacy_model": MODEL_DIR / "spacy_model" / "en_core_web_sm" / "en_core_web_sm-3.8.0",
}

DEFAULT_DATASET = DATA_DIR / "complaints_sample.csv"
```

### **Model Loading** (`models/pipeline_loader.py`)
- **Stub Models**: Auto-fallback for development if models missing
- **Thread Safety**: Pipeline caching with locks for concurrent requests
- **Validation**: Output format checking for all model predictions

### **Rule Thresholds** (`utils/*_rules.py`)
- **Severity Confidence Boost**: 85% for High, 75% for Low, 65% for Medium
- **Urgency Confidence Boost**: 80% for Urgent, 70% for Concerned
- **Keyword Matching**: Case-insensitive substring matching

---

## 💾 Data Schema

### **CSV Output Format** (`data/complaints_sample.csv`)
```csv
created_at,issue_type,severity,urgency,location,complaint_text
2025-11-24T10:30:45Z,Electrical,High,Urgent,Gandhi Nagar,"Dangerous wire..."
```

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `created_at` | ISO 8601 | UTC timestamp | `2025-11-24T10:30:45.123456` |
| `issue_type` | String | Category name | `Electrical`, `Water Crisis` |
| `severity` | Enum | Low/Medium/High | `High` |
| `urgency` | Enum | Neutral/Concerned/Angry/Urgent | `Urgent` |
| `location` | String | Extracted location | `Gandhi Nagar`, `Koramangala` |
| `complaint_text` | String | Original complaint | Full text |

---

## 🎨 UI Components

### **Pages**
1. **Complaint Analyzer**
   - Text input with multi-line support
   - Real-time analysis with spinner
   - Classification summary with color-coded pills
   - Entity extraction with inline annotations
   - Token-level NER visualization
   - Complaint context with highlighted entities
   - Recent analyses history (last 5)

2. **Dashboard & Analytics**
   - Dataset summary metrics (total, today, urgent %)
   - Time-series complaint trends
   - Category distribution bar chart
   - Severity pie chart
   - Severity/Urgency heatmap matrix
   - Urgency timeline area chart
   - Location-based word cloud
   - Latest complaints table (last 20)
   - CSV download button

3. **About**
   - System methodology
   - Technology stack
   - Model training information

### **Sidebar**
- App title and description
- Dark mode toggle
- Snapshot metrics (complaints, urgent %, last update)
- Pipeline component status badges
- Resources section (SDG 9 badge)
- Model information

---

## 🧪 Testing

### **Run All Tests**
```powershell
pytest tests/ -v
```

### **Test Coverage**
- `test_model_loading.py` - Model checkpoint loading, stub fallback
- `test_validation.py` - Output format validation, error handling
- `test_thread_safety.py` - Concurrent pipeline access

### **Manual Testing**
```powershell
# Test with sample data
python -c "from utils.analysis_pipeline import analyze_complaint; from models.pipeline_loader import load_model_bundle; bundle = load_model_bundle(); result = analyze_complaint('Street light not working in Jayanagar', bundle); print(f'Category: {result.issue_type}, Severity: {result.severity}, Urgency: {result.urgency}')"
```

---

## 🛠️ Troubleshooting

### **Common Issues**

**1. ModuleNotFoundError: No module named 'transformers'**
```powershell
pip install transformers torch
```

**2. spaCy model not found**
```powershell
python -m spacy download en_core_web_sm
# Or link existing model
python -m spacy link models/spacy_model/en_core_web_sm/en_core_web_sm-3.8.0 en_core_web_sm
```

**3. CUDA out of memory**
```python
# Edit models/pipeline_loader.py - force CPU
device = torch.device("cpu")
```

**4. Streamlit port already in use**
```powershell
streamlit run app.py --server.port 8502
```

**5. Dataset shows "HIGH" and "High" separately**
```powershell
# Clear and restart - fixed in latest version with .title() normalization
python -c "from pathlib import Path; import pandas as pd; pd.DataFrame(columns=['created_at','issue_type','severity','urgency','location','complaint_text']).to_csv('data/complaints_sample.csv', index=False)"
```

---

## 📈 Performance Metrics

### **Model Inference Speed**
- **Category Prediction**: ~200ms (CPU) / ~50ms (GPU)
- **Severity Prediction**: ~180ms (CPU) / ~45ms (GPU)
- **Urgency Prediction**: ~180ms (CPU) / ~45ms (GPU)
- **NER Extraction**: ~100ms (spaCy)
- **Total Pipeline**: ~500ms average per complaint

### **Memory Usage**
- **Model Loading**: ~800MB (BERT + spaCy)
- **Runtime**: ~1.2GB (with Streamlit overhead)
- **Peak**: ~1.5GB (during batch processing)

### **Accuracy** (on validation set)
- **Category**: 87% accuracy, 0.85 F1-score (weighted)
- **Severity**: 82% accuracy, 0.80 F1-score
- **Urgency**: 79% accuracy, 0.77 F1-score
- **NER**: 91% F1-score (locations)

---

## 🤝 Contributing

Contributions welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Code Style**
- Follow PEP 8 guidelines
- Use type hints for function signatures
- Add docstrings for public functions
- Run `pytest` before committing

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Prashanth** - [@prashanth-31](https://github.com/prashanth-31)

---

## 🙏 Acknowledgments

- **Hugging Face** - Transformers library and model hub
- **spaCy** - Industrial-strength NLP library
- **Streamlit** - Rapid app development framework
- **BBMP** (Bruhat Bengaluru Mahanagara Palike) - Training data source
- **UN SDG 9** - Industry, Innovation and Infrastructure goals

---

## 📞 Support

For issues, questions, or suggestions:
- **GitHub Issues**: [Open an issue](https://github.com/prashanth-31/Smart-Infrastructure-Complaint-Intelligence-System-Using-NLP-LLM-Techniques/issues)
- **Email**: Contact repository owner
- **Documentation**: See code comments and docstrings

---

## 🔮 Roadmap

### **Planned Features**
- [ ] Multi-language support (Hindi, Kannada, Tamil)
- [ ] Image attachment analysis
- [ ] SMS/WhatsApp integration
- [ ] Auto-routing to departments
- [ ] SLA tracking and alerts
- [ ] Mobile app (React Native)
- [ ] RESTful API endpoint
- [ ] Real-time complaint streaming
- [ ] Predictive maintenance models
- [ ] Chatbot interface

### **Model Improvements**
- [ ] Fine-tune on larger municipal datasets
- [ ] Add complaint resolution prediction
- [ ] Implement active learning loop
- [ ] Multi-modal analysis (text + images)
- [ ] Transfer learning for other cities

---

**⭐ Star this repo if you find it useful!**
