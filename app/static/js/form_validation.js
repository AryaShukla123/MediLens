(function () {
  "use strict";

  function initFormValidation(form) {
    const fields = Array.from(form.querySelectorAll(".form-input, .form-select"));

    fields.forEach((field) => {
      field.addEventListener("blur", () => validateField(field));
      field.addEventListener("change", () => validateField(field));
    });

    form.addEventListener("submit", (event) => {
      let firstInvalid = null;

      fields.forEach((field) => {
        const isValid = validateField(field);
        if (!isValid && !firstInvalid) {
          firstInvalid = field;
        }
      });

      if (firstInvalid) {
        event.preventDefault();
        firstInvalid.focus();
      }
    });
  }

  function validateField(field) {
    const errorEl = field
      .closest(".input-group")
      ?.querySelector(`.field-error[data-error-for="${field.name}"]`);

    const message = getValidationMessage(field);

    if (message) {
      field.classList.add("invalid");
      if (errorEl) {
        errorEl.textContent = message;
        errorEl.classList.add("visible");
      }
      return false;
    }

    field.classList.remove("invalid");
    if (errorEl) {
      errorEl.textContent = "";
      errorEl.classList.remove("visible");
    }
    return true;
  }

  function getValidationMessage(field) {
    const value = field.value.trim();

    if (field.hasAttribute("required") && value === "") {
      return "This field is required.";
    }

    if (value === "") {
      return "";
    }

    if (field.tagName === "SELECT") {
      return "";
    }

    if (field.type === "number") {
      const numeric = Number(value);

      if (Number.isNaN(numeric)) {
        return "Enter a valid number.";
      }

      const min = field.getAttribute("min");
      const max = field.getAttribute("max");

      if (min !== null && numeric < Number(min)) {
        return `Must be at least ${min}.`;
      }
      if (max !== null && numeric > Number(max)) {
        return `Must be no more than ${max}.`;
      }
    }

    return "";
  }

  document.addEventListener("DOMContentLoaded", () => {
    document
      .querySelectorAll("form.form-card")
      .forEach((form) => initFormValidation(form));
  });
})();
