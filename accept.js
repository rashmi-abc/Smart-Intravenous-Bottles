function showNotification(event) {
    event.preventDefault(); // Prevent form submission
    alert("Request has been sent successfully!");
    window.location.href = "doctor.html"; // Redirect to doctor.html
}