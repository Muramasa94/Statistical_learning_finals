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

// Function to display sentences with hover functionality
// This will replace the textarea with a list of sentences
function displaySentencesWithHover(perSentenceResults) {
    // Remove old instances of sentence container if exists
    const existingContainer = document.getElementById('sentence-container');
    if (existingContainer) {
        existingContainer.remove();
    }

    // Render sentences in a new container
    const textarea = document.getElementById('inputText');
    const sentences = perSentenceResults.map(r => r.text);
    const sentenceContainer = document.createElement('div');
    sentenceContainer.id = 'sentence-container';

    // Hide input textarea
    textarea.style.display = 'none';
    textarea.parentElement.appendChild(sentenceContainer);

    sentences.forEach((sentence, idx) => {
        const span = document.createElement('span');
        span.textContent = sentence + ' ';
        span.className = 'sentence-hover';
        span.dataset.idx = idx;
        sentenceContainer.appendChild(span);
    });
}

// Tooltip for displaying sentence emotions
const tooltip = document.createElement('div');
tooltip.id = 'sentence-tooltip';
tooltip.style.display = 'none';
tooltip.style.background = '#181825';  // catppuccin-mantle
tooltip.style.color = '#bac2de'; // catppuccin-subtext0
tooltip.style.padding = '8px 12px';
tooltip.style.border = '1px solid #89b4fa'; // catppuccin-blue
tooltip.style.borderRadius = '8px';
tooltip.style.fontSize = '14px';
tooltip.style.zIndex = '9999';
document.body.appendChild(tooltip);

// Setup hover functionality for sentences
function setupSentenceHover(perSentenceResults) {
    const sentenceSpans = document.querySelectorAll('.sentence-hover');
    sentenceSpans.forEach(span => {
        span.addEventListener('mouseenter', function(e) {
            // Highlight
            span.classList.add('highlighted');
            // Get emotions for this sentence
            const idx = span.dataset.idx;
            const emotions = perSentenceResults[idx].emotions;
            tooltip.innerHTML = Object.entries(emotions)
                .map(([emotion, score]) => `<div>${emotion}: ${(score * 100).toFixed(2)}%</div>`)
                .join('');
            tooltip.style.display = 'block';

            // Use Popper.js to position
            Popper.createPopper(span, tooltip, {
                placement: 'top',
                modifiers: [{ name: 'offset', options: { offset: [0, 8] } }]
            });
        });

        span.addEventListener('mouseleave', function(e) {
            span.classList.remove('highlighted');
            tooltip.style.display = 'none';
        });

        // Click a sentence to allow input again, retaining past input
        span.addEventListener('click', function(e) {
            const textarea = document.getElementById('inputText');
            // Show textarea again
            textarea.style.display = '';
            textarea.focus();
            
            // Remove the sentence container:
            const sentenceContainer = document.getElementById('sentence-container');
            if (sentenceContainer) sentenceContainer.remove();

            // Clear tooltip
            tooltip.style.display = 'none';
        });
    });
}

// Receive analysis results from the server
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
    else {
        // Display the results
        displayOverallEmotions(result.overall);

        // Display sentences with hover functionality
        displaySentencesWithHover(result.per_sentence);
        setupSentenceHover(result.per_sentence);
    }

    console.log(result); // For debugging
}