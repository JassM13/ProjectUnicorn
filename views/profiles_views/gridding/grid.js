window.initializeGrid = function(containerId, data) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear any existing grid before initializing a new one
    container.innerHTML = '';

    // Create table structure
    const table = document.createElement('table');
    table.className = 'custom-table';
    
    // Create header
    const headers = ['Name', 'Trades', 'Last Updated', 'Broker Account', 'Actions'];
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    headers.forEach(headerText => {
        const th = document.createElement('th');
        th.textContent = headerText;
        th.addEventListener('click', () => sortTable(table, headers.indexOf(headerText)));
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    tbody.className = 'sortable';
    
    data.forEach(item => {
        const row = document.createElement('tr');
        
        // Add data cells
        row.appendChild(createCell(item.name));
        row.appendChild(createCell(item.trades));
        row.appendChild(createCell(item.last_updated));
        row.appendChild(createCell(item.broker_account ? 'True' : 'False'));
        
        // Add actions cell
        const actionsCell = document.createElement('td');
        actionsCell.className = 'actions-cell';
        actionsCell.appendChild(createActionButton('Edit', '/assets/svgs/Edit/Edit_Pencil.svg'));
        actionsCell.appendChild(createActionButton('Remove', '/assets/svgs/User/User_Remove.svg'));
        row.appendChild(actionsCell);
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    container.appendChild(table);
    
    // Initialize sorting
    makeSortable(tbody);
}

function createCell(content) {
    const td = document.createElement('td');
    td.textContent = content;
    return td;
}

function createActionButton(action, iconPath) {
    const button = document.createElement('button');
    button.className = 'action-button';
    
    const img = document.createElement('img');
    img.src = iconPath;
    img.alt = action;
    
    button.appendChild(img);
    button.addEventListener('click', () => handleAction(action));
    return button;
}

function handleAction(action) {
    console.log(`${action} clicked`);
    // Implement action handling here
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const headers = table.querySelectorAll('th');
    
    // Toggle sort direction
    const header = headers[columnIndex];
    const isAscending = header.classList.toggle('asc');
    header.classList.toggle('desc', !isAscending);
    
    // Remove sort classes from other headers
    headers.forEach((h, i) => {
        if (i !== columnIndex) {
            h.classList.remove('asc', 'desc');
        }
    });
    
    // Sort rows
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent;
        const bValue = b.cells[columnIndex].textContent;
        
        if (!isNaN(aValue) && !isNaN(bValue)) {
            return isAscending ? aValue - bValue : bValue - aValue;
        }
        
        return isAscending 
            ? aValue.localeCompare(bValue)
            : bValue.localeCompare(aValue);
    });
    
    // Reorder rows
    rows.forEach(row => tbody.appendChild(row));
}

function makeSortable(tbody) {
    let draggedRow = null;
    
    tbody.addEventListener('dragstart', e => {
        draggedRow = e.target.closest('tr');
        e.target.style.opacity = '0.5';
    });
    
    tbody.addEventListener('dragend', e => {
        e.target.style.opacity = '';
    });
    
    tbody.addEventListener('dragover', e => {
        e.preventDefault();
        const row = e.target.closest('tr');
        if (row && row !== draggedRow) {
            const rect = row.getBoundingClientRect();
            const mid = (rect.top + rect.bottom) / 2;
            if (e.clientY < mid) {
                row.parentNode.insertBefore(draggedRow, row);
            } else {
                row.parentNode.insertBefore(draggedRow, row.nextSibling);
            }
        }
    });
}