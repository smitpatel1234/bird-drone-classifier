document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const fileSelectBtn = document.getElementById('file-select-btn');
    const fileName = document.getElementById('file-name');
    const loadingIndicator = document.querySelector('.loading');
    const resultContainer = document.getElementById('result-container');
    const resultIcon = document.getElementById('result-icon');
    const resultTitle = document.getElementById('result-title');
    const resultDescription = document.getElementById('result-description');
    const confidenceScore = document.getElementById('confidence-score');
    const confidenceProgress = document.getElementById('confidence-progress');
    const downloadReportBtn = document.getElementById('download-report-btn');
    const uploadNewBtn = document.getElementById('upload-new-btn');
    
    let currentFile = null;
    let lastPrediction = null;
    let lastConfidence = null;
    let reportId = null;
    
    // Handle file selection button
    fileSelectBtn.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });
    
    // Handle drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#007bff';
        uploadArea.style.backgroundColor = '#e9f7fe';
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.style.borderColor = '#ced4da';
        uploadArea.style.backgroundColor = '';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#ced4da';
        uploadArea.style.backgroundColor = '';
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Handle file processing
    function handleFile(file) {
        // Check if file is a .pkl file
        if (!file.name.toLowerCase().endsWith('.pkl')) {
            alert('Please upload a .pkl file containing radar signature data');
            return;
        }
        
        currentFile = file;
        fileName.textContent = file.name;
        
        // Show loading indicator
        uploadArea.style.display = 'none';
        loadingIndicator.style.display = 'block';
        
        // Create form data for file upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Send file to server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loadingIndicator.style.display = 'none';
            
            if (data.error) {
                alert('Error: ' + data.error);
                uploadArea.style.display = 'block';
                return;
            }
            
            // Display result
            showResult(data);
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            uploadArea.style.display = 'block';
            alert('Error uploading file: ' + error);
        });
    }
    
    // Display result
    function showResult(data) {
        lastPrediction = data.prediction;
        lastConfidence = data.confidence;
        reportId = data.report_id;
        
        resultTitle.textContent = `Classification Result: ${data.prediction}`;
        
        if (data.prediction === 'Bird') {
            resultContainer.className = 'result-container bird-result';
            resultIcon.innerHTML = '<span style="color: green;">🐦</span>';
            resultDescription.textContent = 'The radar signature in the uploaded file has been classified as a bird.';
        } else {
            resultContainer.className = 'result-container drone-result';
            resultIcon.innerHTML = '<span style="color: red;">🚁</span>';
            resultDescription.textContent = 'The radar signature in the uploaded file has been classified as a drone.';
        }
        
        // Update confidence score
        if (data.confidence !== 'Not available') {
            const confidenceValue = parseFloat(data.confidence);
            confidenceScore.textContent = data.confidence;
            confidenceProgress.style.width = data.confidence;
            
            if (confidenceValue < 70) {
                confidenceProgress.className = 'progress-bar bg-warning';
            } else {
                confidenceProgress.className = 'progress-bar bg-success';
            }
        } else {
            confidenceScore.textContent = 'Not available';
            confidenceProgress.style.width = '0%';
        }
        
        resultContainer.style.display = 'block';
    }
    
    // Handle download report button
    downloadReportBtn.addEventListener('click', function() {
        if (!reportId) return;
        
        const url = `/download_report/${reportId}?filename=${encodeURIComponent(currentFile.name)}&prediction=${encodeURIComponent(lastPrediction)}&confidence=${encodeURIComponent(lastConfidence)}`;
        window.location.href = url;
    });
    
    // Handle upload new file button
    uploadNewBtn.addEventListener('click', function() {
        resultContainer.style.display = 'none';
        uploadArea.style.display = 'block';
        fileName.textContent = '';
        currentFile = null;
        lastPrediction = null;
        lastConfidence = null;
        reportId = null;
        fileInput.value = '';
    });
});
