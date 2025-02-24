// Get the anchor elements
const homeLink = document.querySelector(".nav ul li:nth-child(1) a");
const loginLink = document.querySelector(".nav ul li:nth-child(2) a");
const registrationLink = document.querySelector(".nav ul li:nth-child(3) a");

// Get the content elements
const homeContent = document.querySelector(".content-home");
const loginContent = document.querySelector(".content-login");
const registrationContent = document.querySelector(".content-registration");

// Add click event listeners
homeLink.addEventListener("click", () => {
    homeContent.style.display = "block";
    loginContent.style.display = "none";
    registrationContent.style.display = "none";
});

loginLink.addEventListener("click", () => {
    homeContent.style.display = "none";
    loginContent.style.display = "block";
    registrationContent.style.display = "none";
});

registrationLink.addEventListener("click", () => {
    homeContent.style.display = "none";
    loginContent.style.display = "none";
    registrationContent.style.display = "block";
});
