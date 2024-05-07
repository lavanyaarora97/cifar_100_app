let selectedFiles = [];
let imgs = ''
function handleFiles(files) {
    selectedFiles = selectedFiles.concat(Array.from(files)); // Combine old and new files
    const fileList = document.getElementById("file-list");
    fileList.innerHTML = ''; // Clear the list to redraw it
    
const annotatedImage = document.getElementById('annotatedImage');
console.log(annotatedImage)
    selectedFiles.forEach(file => {
        const li = document.createElement("div");
        const image = document.createElement("img");
        image.src = URL.createObjectURL(file); // Set the image source to the file URL
        image.style.width = '300px'; // Set max width to keep images within a reasonable size
        image.style.height = '300px'; // Set max height to keep images within a reasonable size
        
        li.style.height ='300px'
        li.appendChild(image); // Append the image to the list item
        fileList.appendChild(li); // Append the list item to the file list
    });
    // document.getElementById("classify-btn").style.display = 'block'; // Make sure this ID matches your submit button for classification
    // document.getElementById("detect-btn").style.display = 'block';  // Make sure this ID matches your submit button for detection

    console.log(selectedFiles[0])
    uploadImage()

    
}

async function uploadImage() {
    console.log(1)
    const formData = new FormData();
    //const fileInput = document.getElementById('uploadInput');
    formData.append('file', selectedFiles[0]);

    const response = await fetch("https://cifar100-image-classification-backend-api.onrender.com/upload", {
        method: 'POST',
        body: formData
    });

    const responseData = await response.json();

    console.log(responseData)

    const results = document.getElementById('detection-results');
    results.innerHTML = ''; // Clear previous results

    if(responseData){
    responseData.forEach(detections => {
        const detectionDiv = document.createElement('div');
        detectionDiv.textContent = 'Detected Objects:';
        
            const detItem = document.createElement('p');
            detItem.textContent = `Object Class: ${detections.class}, Confidence: ${detections.confidence.toFixed(2)}%`;
            detectionDiv.appendChild(detItem);
            
            detItem.style.backgroundColor = `${detections.css}`;
            detItem.style.color = 'white';
            detItem.style.borderRadius = '5px'; // Set border radius to 5px
            detItem.style.padding = '10px'; // Set padding to 10px


        
        results.appendChild(detectionDiv);
        selectedFiles = []
    });
     // Display annotated image
     if (responseData.annotated_image) {
        
        annotatedImage.src = `${selectedFiles[0]}`;
        console.log('show')
        annotatedImage.style.display = 'block';
        
     }
    
}

    // Display detection results
    // const resultDiv = document.getElementById('result');
    // resultDiv.innerHTML = '<h2>Detection Results:</h2>';
    // responseData.detections.forEach(det => {
    //     resultDiv.innerHTML += `<p>Class: ${det.class}, Confidence: ${det.confidence}</p>`;
    // });

  

}

document.getElementById('drop-area').addEventListener('dragover', (event) => {
    event.stopPropagation();
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    event.target.style.backgroundColor = '#f0f0f0'; // Highlight color on drag over
});

document.getElementById('drop-area').addEventListener('dragleave', (event) => {
    event.target.style.backgroundColor = 'transparent'; // Revert color on drag leave
});

document.getElementById('drop-area').addEventListener('drop', (event) => {
    event.stopPropagation();
    event.preventDefault();
    const files = event.dataTransfer.files;
    handleFiles(files);
    event.target.style.backgroundColor = 'transparent'; // Revert color after dropping
});

document.querySelector('form').onsubmit = function(e) {
    e.preventDefault();
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('file', file);
    });

    fetch("http://127.0.0.1:10000/upload", {  // Use the current path for the POST request
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
       console.log(data)
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error processing your request. Please try again.');
    });
};