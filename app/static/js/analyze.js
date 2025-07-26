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

    console.log(result); // For debugging
}