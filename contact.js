// JavaScript to handle form submission
document.querySelector("form").addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent default form submission

    // Get input values
    const name = document.querySelector("input[name='name']").value.trim();
    const email = document.querySelector("input[name='email']").value.trim();
    const phone = document.querySelector("input[name='phone']").value.trim();
    const message = document.querySelector("textarea[name='message']").value.trim();

    // Validate inputs
    if (!name || !email || !phone || !message) {
        alert("Please fill in all the fields before sending your message.");
        return; // Do not proceed if validation fails
    }

    // If all fields are filled, show success message and redirect
    alert("Thanks for contacting us. You will receive an email for further clarification.");
    window.location.href = "home.html"; // Change "suspended.html" to the appropriate page
});
const inputs = document.querySelectorAll(".input");

function focusFunc() {
  let parent = this.parentNode;
  parent.classList.add("focus");
}

function blurFunc() {
  let parent = this.parentNode;
  if (this.value == "") {
    parent.classList.remove("focus");
  }
}

inputs.forEach((input) => {
  input.addEventListener("focus", focusFunc);
  input.addEventListener("blur", blurFunc);
});