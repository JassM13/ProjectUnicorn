function createGridTable(data) {
    if (!data.length) {
        return ''; // Return an empty string if no data is available
    }

    const headers = ['Name', 'Trades', 'Last Updated', 'Broker Account', 'Actions'];
    const cellStyle = "padding: 8px 16px; background-color: #000; color: #fff; text-align: left;";

    let tableHtml = `
    <table class="custom-table" style="width: 100%; border-collapse: separate; border-spacing: 0; border-radius: 8px; overflow: hidden; background: rgba(26, 26, 26, 0.7); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);">
        <thead>
            <tr>
                ${headers.map(header => `<th style="
                    padding: 16px 16px;
                    background: rgba(255, 255, 255, 0.05);
                    color: #fff;
                    text-align: left;
                ">${header}</th>`).join('')}
            </tr>
        </thead>
        <tbody class="sortable">
            ${data.map(item => `
                <tr>
                    <td style='${cellStyle}'>${item.name || ''}</td>
                    <td style='${cellStyle}'>${item.trades || 0}</td>
                    <td style='${cellStyle}'>${item.last_updated || ''}</td>
                    <td style='${cellStyle}'>${item.broker_account ? 'True' : 'False'}</td>
                    <td style='${cellStyle}'>
                        <button style="background: none; border: none; cursor: pointer; padding: 4px; filter: brightness(0) saturate(100%) invert(91%) sepia(9%) saturate(2661%) hue-rotate(335deg) brightness(60%) contrast(80%);"
                                onmouseover="this.style.filter='brightness(0) saturate(100%) invert(91%) sepia(9%) saturate(2661%) hue-rotate(335deg) brightness(99%) contrast(80%)'"
                                onmouseout="this.style.filter='brightness(0) saturate(100%) invert(91%) sepia(9%) saturate(2661%) hue-rotate(335deg) brightness(60%) contrast(80%)'">
                            <img src='/assets/svgs/Edit/Edit_Pencil.svg' alt='Edit'>
                        </button>
                        <button style="background: none; border: none; cursor: pointer; padding: 4px; filter: brightness(0) saturate(80%) invert(16%) sepia(99%) saturate(7444%) hue-rotate(359deg) brightness(40%) contrast(60%);"
                                onmouseover="this.style.filter='brightness(0) saturate(80%) invert(16%) sepia(99%) saturate(7444%) hue-rotate(359deg) brightness(60%) contrast(80%)'"
                                onmouseout="this.style.filter='brightness(0) saturate(80%) invert(16%) sepia(99%) saturate(7444%) hue-rotate(359deg) brightness(40%) contrast(60%)'"
                                @click="showDeleteDialog(this, '${item.name || 'this profile'}', '${item.id}')"> <!-- Enclose item.id in quotes -->
                            <img src='/assets/svgs/User/User_Remove.svg' alt='Remove'>
                        </button>
                    </td>
                </tr>
            `).join('')}
        </tbody>
    </table>
    `;
    
    return tableHtml;
}

function showDeleteDialog(button, profileName, profileId) {
    const dialog = document.createElement('div');
    dialog.style.cssText = 'position: fixed; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(0, 0, 0, 0.7); display: flex; justify-content: center; align-items: center; z-index: 1000; opacity: 0; visibility: hidden; transition: opacity 0.3s ease, visibility 0.3s ease;';
    
    const content = document.createElement('div');
    content.style.cssText = 'background: #111; padding: 24px; border-radius: 16px; width: 90%; max-width: 500px; position: fixed; left: 50%; top: 50%; transform: translate(-50%, -50%) scale(0.8); transition: all 0.3s cubic-bezier(0.68, -0.55, 0.27, 1.55); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3); opacity: 0; visibility: hidden;';

    content.innerHTML = `
        <h2 style="margin: 0 0 20px 0; color: #f6cd70;">Delete Profile</h2>
        <p style="margin: 0 0 24px 0; color: white;">Are you sure you want to delete "${profileName}"?</p>
        <div style="display: flex; justify-content: flex-end; gap: 12px;">
            <button class="cancel" style="padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; background: #333; color: white; transition: all 0.3s ease;">Cancel</button>
            <button class="confirm" style="padding: 12px 24px; border: none; border-radius: 8px; cursor: pointer; font-weight: 800; background: #f6cd70; color: black; transition: all 0.3s ease;"
            @click="fetch('/api/profiles/delete/${profileId}', { 
                    method: 'DELETE'
                })
                .then(res => res.json())
                .then(data => {
                    console.log('Profile deleted:', data);
                })
                .catch(error => console.error('Error:', error));"
            >Delete</button>
        </div>
    `;
    
    dialog.appendChild(content);
    document.body.appendChild(dialog);
    
    // Trigger reflow to ensure transitions work
    dialog.offsetHeight;
    dialog.style.opacity = '1';
    dialog.style.visibility = 'visible';
    content.style.opacity = '1';
    content.style.visibility = 'visible';
    content.style.transform = 'translate(-50%, -50%) scale(1)';
    
    const closeDialog = () => {
        dialog.style.opacity = '0';
        dialog.style.visibility = 'hidden';
        content.style.opacity = '0';
        content.style.visibility = 'hidden';
        content.style.transform = 'translate(-50%, -50%) scale(0.8)';
        setTimeout(() => document.body.removeChild(dialog), 300);
    };
    
    dialog.querySelector('.cancel').onclick = closeDialog;
    dialog.querySelector('.confirm').onclick = () => {
        htmx.trigger(button, 'confirmed');
        closeDialog();
    };
    
    dialog.onclick = (e) => {
        if (e.target === dialog) closeDialog();
    };
    
    // Add hover effect to buttons
    const buttons = dialog.querySelectorAll('button');
    buttons.forEach(button => {
        button.onmouseover = () => button.style.transform = 'translateY(-1px)';
        button.onmouseout = () => button.style.transform = 'none';
    });
}