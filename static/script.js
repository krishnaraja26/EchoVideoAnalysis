// Get the sidebar element by its ID
const sidebar = document.getElementById('sidebar');

// When mouse enters the sidebar, expand it
sidebar.addEventListener('mouseenter', () => {
    sidebar.style.width = '300px';  // Expands the sidebar to 300px width
});

// When mouse leaves the sidebar, collapse it
sidebar.addEventListener('mouseleave', () => {
    sidebar.style.width = '250px';  // Resets the sidebar width to 250px
});
