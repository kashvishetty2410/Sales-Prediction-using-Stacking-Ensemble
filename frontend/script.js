// Sales Prediction System - JavaScript

const API_URL = 'http://localhost:8000';

// DOM Elements
const form = document.getElementById('predictionForm');
const resultDiv = document.getElementById('result');
const errorDiv = document.getElementById('error');
const loadingDiv = document.getElementById('loading');
const salesValue = document.getElementById('salesValue');
const errorMessage = document.getElementById('errorMessage');

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results
    resultDiv.style.display = 'none';
    errorDiv.style.display = 'none';
    
    // Show loading
    loadingDiv.style.display = 'block';
    
    // Get form data
    const formData = {
        ship_mode: document.getElementById('shipMode').value,
        segment: document.getElementById('segment').value,
        category: document.getElementById('category').value,
        sub_category: document.getElementById('subCategory').value
    };
    
    try {
        // Make API request
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display result
        salesValue.textContent = `$${data.predicted_sales.toFixed(2)}`;
        resultDiv.style.display = 'block';
        
    } catch (error) {
        // Display error
        errorMessage.textContent = error.message || 'Failed to get prediction';
        errorDiv.style.display = 'block';
        console.error('Error:', error);
    } finally {
        // Hide loading
        loadingDiv.style.display = 'none';
    }
});

// Check API health on page load
async function checkAPI() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('API is running');
        }
    } catch (error) {
        console.warn('API not running. Make sure to start the backend with: uvicorn main:app --reload');
    }
}

// Initialize
checkAPI();
