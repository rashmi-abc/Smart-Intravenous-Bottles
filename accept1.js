// Freeze checkboxes for Medications
document.querySelector('.update-btn-med').addEventListener('click', () => {
  const medicationCheckboxes = document.querySelectorAll('.med-checkbox');
  
  medicationCheckboxes.forEach((checkbox) => {
    checkbox.disabled = true; // Disable all checkboxes in the Medications section
  });
  
  alert('Medications updated successfully!');
});

// Freeze checkboxes for Infusions
document.querySelector('.update-btn-inf').addEventListener('click', () => {
  const infusionCheckboxes = document.querySelectorAll('.inf-checkbox');
  
  infusionCheckboxes.forEach((checkbox) => {
    checkbox.disabled = true; // Disable all checkboxes in the Infusions section
  });
  
  alert('Infusions updated successfully!');
});

// Freeze checkboxes for Injections
document.querySelector('.update-btn-inj').addEventListener('click', () => {
  const injectionCheckboxes = document.querySelectorAll('.inj-checkbox');
  
  injectionCheckboxes.forEach((checkbox) => {
    checkbox.disabled = true; // Disable all checkboxes in the Injections section
  });
  
  alert('Injections updated successfully!');
});
