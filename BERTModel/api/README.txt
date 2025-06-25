# DeepSNP3D API

A FastAPI-based REST API for DNA sequence analysis using a BERT-based deep learning model.

## Features

- DNA sequence validation and analysis
- BERT-based feature extraction
- GPU/CPU support with automatic device detection
- Interactive API documentation
- CORS support for web integration
- Health monitoring endpoints

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure your project structure matches:
   ```
   DeepSNP3D/
   ├── BERTModel/
   │   ├── api/
   │   │   ├── api.py
   │   │   ├── run_server.py
   │   │   └── requirements.txt
   │   └── model/
   │       ├── Model.py
   │       └── DNABERT-2-117M/
   └── DeepSNP3D.pth (trained model weights)
   ```

## Running the API

### Development Mode

From the DeepSNP3D root directory:

```bash
# Navigate to project root
cd c:\Users\lhfor\AppData\Local\Programs\Python\Python38-32\DeepSNP3D

# Run with uvicorn directly
uvicorn BERTModel.api.main:app --reload --host 0.0.0.0 --port 8000

# Or run the Python script
python -m BERTModel.api.main
```

### Production Mode

Using the run_server.py script:

```bash
# From the api directory
cd BERTModel/api
python run_server.py --workers 4 --port 8000

# Or with custom options
python run_server.py --host 0.0.0.0 --port 8080 --workers 2 --log-level info
```

## API Endpoints

### Core Endpoints

- **GET /** - Root endpoint with API information
- **GET /health** - Health check and model status
- **POST /predict** - Main prediction endpoint for DNA sequences
- **GET /model-info** - Information about the loaded model
- **GET /docs** - Interactive Swagger UI documentation
- **GET /redoc** - ReDoc API documentation

### Example Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### DNA Sequence Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sequence": "ATCGATCGATCG"}'
```

#### Using Python requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict sequence features
data = {"sequence": "ATCGATCGATCGATCG"}
response = requests.post("http://localhost:8000/predict", json=data)
features = response.json()
print(features)
```

## Web Integration

### JavaScript/Vanilla JS

```javascript
// Health check
fetch('http://localhost:8000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Predict DNA sequence
const predictSequence = async (sequence) => {
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ sequence: sequence })
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return await response.json();
};

// Usage
predictSequence('ATCGATCGATCG')
  .then(result => {
    console.log('Features:', result.features);
    // Process features for visualization
  })
  .catch(error => console.error('Error:', error));
```

### React Integration

```jsx
import React, { useState, useEffect } from 'react';

const DNAAnalyzer = () => {
  const [sequence, setSequence] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiHealth, setApiHealth] = useState(null);

  // Check API health on component mount
  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => setApiHealth(data))
      .catch(err => console.error('API health check failed:', err));
  }, []);

  const analyzeSequence = async () => {
    if (!sequence.trim()) return;
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: sequence.toUpperCase() })
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
      alert(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>DNA Sequence Analyzer</h2>
      
      {/* API Status */}
      <div style={{ 
        padding: '10px', 
        backgroundColor: apiHealth?.status === 'healthy' ? '#d4edda' : '#f8d7da',
        marginBottom: '20px'
      }}>
        API Status: {apiHealth?.status || 'Unknown'} 
        {apiHealth?.device && ` (Device: ${apiHealth.device})`}
      </div>

      {/* Input */}
      <div>
        <textarea
          value={sequence}
          onChange={(e) => setSequence(e.target.value)}
          placeholder="Enter DNA sequence (A, T, C, G, N)"
          rows={4}
          cols={50}
          style={{ marginBottom: '10px' }}
        />
        <br />
        <button onClick={analyzeSequence} disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Sequence'}
        </button>
      </div>

      {/* Results */}
      {result && (
        <div style={{ marginTop: '20px' }}>
          <h3>Analysis Results</h3>
          <p><strong>Sequence Length:</strong> {result.sequence_length}</p>
          <p><strong>Model Version:</strong> {result.model_version}</p>
          
          {/* Feature visualization would go here */}
          <details>
            <summary>Raw Features (Click to expand)</summary>
            <pre>{JSON.stringify(result.features, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
};

export default DNAAnalyzer;
```

### D3.js Visualization

```javascript
// Example D3.js integration for feature visualization
const visualizeFeatures = (features) => {
  // Clear previous visualization
  d3.select("#visualization").selectAll("*").remove();
  
  const svg = d3.select("#visualization")
    .append("svg")
    .attr("width", 800)
    .attr("height", 400);

  // Example: Visualize attention weights
  if (features.attention_weights) {
    const data = features.attention_weights[0]; // First head
    
    const heatmap = svg.selectAll(".cell")
      .data(data.flat())
      .enter()
      .append("rect")
      .attr("class", "cell")
      .attr("x", (d, i) => (i % data[0].length) * 20)
      .attr("y", (d, i) => Math.floor(i / data[0].length) * 20)
      .attr("width", 18)
      .attr("height", 18)
      .attr("fill", d => d3.interpolateViridis(d));
  }

  // Example: Feature importance bar chart
  if (features.feature_importance) {
    const margin = {top: 20, right: 30, bottom: 40, left: 90};
    const width = 800 - margin.left - margin.right;
    const height = 400 - margin.top - margin.bottom;

    const x = d3.scaleLinear()
      .domain([0, d3.max(features.feature_importance)])
      .range([0, width]);

    const y = d3.scaleBand()
      .domain(features.feature_importance.map((d, i) => `Feature ${i}`))
      .range([0, height])
      .padding(0.1);

    const chart = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    chart.selectAll(".bar")
      .data(features.feature_importance)
      .enter().append("rect")
      .attr("class", "bar")
      .attr("x", 0)
      .attr("y", (d, i) => y(`Feature ${i}`))
      .attr("width", d => x(d))
      .attr("height", y.bandwidth())
      .attr("fill", "steelblue");
  }
};

// Usage after getting prediction results
predictSequence('ATCGATCGATCG')
  .then(result => {
    visualizeFeatures(result.features);
  });
```

### Express.js Backend Integration

```javascript
const express = require('express');
const axios = require('axios');
const app = express();

app.use(express.json());

// Proxy endpoint to your Python API
app.post('/api/analyze-dna', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:8000/predict', {
      sequence: req.body.sequence
    });
    
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ 
      error: 'Analysis failed', 
      detail: error.response?.data?.detail || error.message 
    });
  }
});

app.listen(3000, () => {
  console.log('Frontend server running on http://localhost:3000');
});
```

## Configuration

### Environment Variables

Create a `.env` file:
```
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=C:\Users\lhfor\AppData\Local\Programs\Python\Python38-32\DeepSNP3D.pth
MAX_SEQUENCE_LENGTH=384
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Production Deployment

For production deployment, consider:

1. **Use a production WSGI server** like Gunicorn:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker BERTModel.api.api:app
   ```

2. **Configure CORS properly** - replace `allow_origins=["*"]` with specific domains

3. **Add authentication** if needed

4. **Use environment variables** for sensitive configuration

5. **Set up monitoring** and logging

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'BERTModel'**
   - Run from the DeepSNP3D root directory
   - Set PYTHONPATH if needed: `set PYTHONPATH=%CD%`

2. **Model not found errors**
   - Ensure DNABERT-2-117M folder exists in BERTModel/model/
   - Check DeepSNP3D.pth file location

3. **CUDA out of memory**
   - Reduce max_length in tokenizer
   - Use CPU instead: set CUDA_VISIBLE_DEVICES=""

4. **CORS errors in browser**
   - API includes CORS middleware
   - For production, configure specific origins

### API Testing

Use the interactive documentation at `http://localhost:8000/docs` to test endpoints directly in your browser.

## Support

For issues or questions, check the FastAPI documentation