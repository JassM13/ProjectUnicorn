async function createDropdown(profiles) {
    if (!profiles.length) {
        console.log("No profiles found");
        return '';
    }
    console.log("Profiles:", profiles);
    const dropdownHtml = `
        <select
            x-model="selectedProfile"
            @change="htmx.trigger('#tradesGrid', 'htmx:refresh', {profile: selectedProfile})"
            style="background-color: #1a1a1a; color: white; border: 1px solid rgba(255, 255, 255, 0.1);
                   border-radius: 16px; padding: 8px 16px; font-size: 14px; margin-right: 12px;"
        >
            <option value="" selected>All Profiles</option>
            ${profiles.map(profile => `
                <option value="${profile.id}">${profile.name}</option>
            `).join('')}
        </select>
    `;
    return dropdownHtml;
}