// header js
const header = document.querySelector("header");

window.addEventListener ("scroll", function() {
    header.classList.toggle ("sticky", this.window.scrollY > 200)
});

// select box function
const selectBtns = document.querySelectorAll(".select-btn");

selectBtns.forEach(item => {
	let itemParent = item.parentElement;
	item.addEventListener("click", () => {
		itemParent.classList.toggle("active");
	});

    let btnText = item.querySelector("span")
    let selectOptions = itemParent.querySelector(".select-options")
    let options = selectOptions.querySelectorAll("li")
    let inputField = selectOptions.parentElement.querySelector(".text-input");

    options.forEach(option => {
        option.addEventListener("click", () => {
            inputField.style.display = "block";
            inputField.value = option.getAttribute("id");
            btnText.innerHTML = option.innerHTML;
            itemParent.classList.toggle("active");
        })
    })
});




