document.getElementById('search-form').addEventListener('submit', function (event) {
    event.preventDefault();
    
    let query = document.getElementById('query').value;
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '';

    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'query': query
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        displayResults(data);
        displayChart(data);
    });
});

function displayResults(data) {
    let resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<h2>Results</h2>';
    for (let i = 0; i < data.documents.length; i++) {
        let docDiv = document.createElement('div');
        docDiv.innerHTML = `<strong>Document ${data.indices[i]}</strong><p>${data.documents[i]}</p><br><strong>Similarity: ${data.similarities[i]}</strong>`;
        resultsDiv.appendChild(docDiv);
    }
}

let chartInstance = null;
function displayChart(data) {
    // Input: data (object) - contains the following keys:
    //        - documents (list) - list of documents
    //        - indices (list) - list of indices   
    //        - similarities (list) - list of similarities
    // TODO: Implement function to display chart here
    //       There is a canvas element in the HTML file with the id 'similarity-chart'
    var indices = data.indices;
    var similarities = data.similarities;

    const ctx = document.getElementById('similarity-chart').getContext('2d');

    if (chartInstance !== null) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: indices.map(String),
            datasets: [{
                label: 'Cosine SImularity',
                data: similarities,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1 
            }]
        },
        options: {
            scales: {
              y: {
                beginAtZero: true
              }
            }
          }
    });
}