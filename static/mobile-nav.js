// mobile-navlist js
const mobileNavBtn = document.querySelector(".mobile-navlist-btn");
const mobileNav = document.querySelector(".navlist");

// open nav state

mobileNavBtn.addEventListener("click", function() {
    const isVisible = mobileNav.getAttribute("data-visible");
    if (isVisible === "false"){
    mobileNav.setAttribute("data-visible", true);
    document.body.style.overflowY = "hidden";
    }

    else{
        mobileNav.setAttribute("data-visible", false);
        document.body.style.overflowY = "auto"; 

    }
});