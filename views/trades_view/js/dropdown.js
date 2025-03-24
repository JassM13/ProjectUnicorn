async function createDropdown(profiles) {
    if (!profiles.length) {
        console.log("No profiles found");
        return `
            <select
                disabled
                style="background-color: #1a1a1a; color: #666; border: 1px solid rgba(255, 255, 255, 0.1);
                       border-radius: 16px; font-size: 14px; font-weight: 600;
                       padding: 8px 12px; min-width: 180px; cursor: not-allowed;
                       height: 40px;"
            >
                <option>Loading profiles...</option>
            </select>
        `;
    }
    console.log("Profiles:", profiles);
    const dropdownHtml = `
        <select
            x-model="selectedProfile"
            @change="htmx.trigger('#tradesGrid', 'htmx:refresh', {profile: selectedProfile})"
            style="background-color: #1a1a1a; color: white; border: 1px solid rgba(255, 255, 255, 0.1);
                   border-radius: 16px; font-size: 14px; font-weight: 600;
                   padding: 8px 12px; min-width: 180px; height: 40px;"
        >
            <option value="" selected>All Profiles</option>
            ${profiles.map(profile => `
                <option value="${profile.id}">${profile.name}</option>
            `).join('')}
        </select>
    `;
    return dropdownHtml;
}
