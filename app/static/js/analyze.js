function displayOverallEmotions(overall) {
    const resultSection = document.querySelector('.card-body');
    resultSection.innerHTML = ''; // Clear previous results

    if (overall && Object.keys(overall).length > 0) {
        const title = document.createElement('h6');
        title.className = 'card-title text-center mb-3';
        title.textContent = 'Overall Emotions';
        resultSection.appendChild(title);

        const ul = document.createElement('ul');
        ul.className = 'list-group list-group-flush';

        for (const [emotion, score] of Object.entries(overall)) {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex justify-content-between align-items-center';
            li.textContent = emotion;

            const span = document.createElement('span');
            span.className = 'badge rounded-pill';
            span.textContent = (score * 100).toFixed(2) + ' %'; // Convert to percentage

            li.appendChild(span);
            ul.appendChild(li);
        }
        resultSection.appendChild(ul);
    } else {
        const p = document.createElement('p');
        p.className = 'text-muted text-center';
        p.textContent = 'Results will appear here after analysis.';
        resultSection.appendChild(p);
    }
}

async function analyzeText() {
    const textarea = document.getElementById('inputText');
    const text = textarea.value;

    const response = await fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text })
    });

    const result = await response.json();

    // Check if the response is successful
    if (!response.ok) {
        // Show error popup
        Swal.fire({
            icon: 'error',
            title: 'Oops...',
            text: result.error || 'An error occurred!',
        });
        return;
    }
    else { // Display the results
        displayOverallEmotions(result.overall);
    }

    console.log(result); // For debugging
}