function createStockCard(stock) {
    const card = document.createElement('div');
    card.className = 'card text-white m-2 position-relative rounded-0';
    card.id = `stock-${stock.symbol}`;
    card.dataset.initialPrice = stock.current_price;
    card.dataset.dayHigh = stock.day_high;
    card.dataset.dayLow = stock.day_low;

    // Fixed square size
    card.style.width = '150px';
    card.style.height = '150px';

    card.innerHTML = `
        <button class="btn position-absolute top-0 end-0 m-1 p-0 text-white hover-red"
                onclick="this.closest('.card').remove()">
            &times;
        </button>

        <div class="card-body text-center p-2 d-flex flex-column justify-content-center align-items-center h-100">
            <h5 class="card-title mb-1">${stock.symbol}</h5>
            <p class="card-text mb-1 small" data-last-price="${stock.current_price}">
                $${stock.current_price.toFixed(2)} ${stock.currency}
            </p>
            <p class="percentage-change text-white small mb-1"></p>
            <p class="daily-range text-white small mb-0">
                H: <span class="day-high">${stock.day_high.toFixed(2)}</span> 
                L: <span class="day-low">${stock.day_low.toFixed(2)}</span>
            </p>
        </div>
    `;
    return card;
}
