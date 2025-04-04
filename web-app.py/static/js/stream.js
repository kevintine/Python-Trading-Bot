document.addEventListener('DOMContentLoaded', function() {
    const socket = io();
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const symbolSelect = document.getElementById('symbolSelect');
    const tradeData = document.getElementById('tradeData');
    let selectedSymbols = [];

    // Handle incoming trade data
    socket.on('trade_update', function(data) {
        const tradeValue = (data.price * data.size).toFixed(2);
        const newRow = document.createElement('tr');
        newRow.className = 'trade-row new';
        newRow.innerHTML = `
            <td><strong>${data.symbol}</strong></td>
            <td>$${data.price.toFixed(2)}</td>
            <td><span class="badge rounded-pill bg-${data.size > 1000 ? 'success' : data.size > 500 ? 'info' : 'secondary'} badge-volume">${data.size}</span></td>
            <td>${new Date(data.timestamp).toLocaleTimeString()}</td>
            <td>$${tradeValue}</td>
        `;
        
        // Add to top of table
        tradeData.prepend(newRow);
        
        // Remove highlight after 1 second
        setTimeout(() => {
            newRow.classList.remove('new');
        }, 1000);

        // Keep only the last 100 entries
        if (tradeData.children.length > 100) {
            tradeData.removeChild(tradeData.lastChild);
        }
    });

    // Start stream button
    startBtn.addEventListener('click', function() {
        selectedSymbols = Array.from(symbolSelect.selectedOptions)
            .map(option => option.value);
        
        if (selectedSymbols.length === 0) {
            alert('Please select at least one stock');
            return;
        }

        fetch('/start_stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbols: selectedSymbols })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                startBtn.disabled = true;
                stopBtn.disabled = false;
                tradeData.innerHTML = ''; // Clear previous data
            }
        });
    });

    // Stop stream button
    stopBtn.addEventListener('click', function() {
        fetch('/stop_stream')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'stopped') {
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        });
    });
});