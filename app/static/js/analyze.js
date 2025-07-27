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

    console.log(result); // For debugging
}