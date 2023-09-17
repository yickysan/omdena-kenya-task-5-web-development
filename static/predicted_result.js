const container = document.querySelector(".predict");
let resultContainer = container.querySelector(".result-container")

let text1 = "Serious Injury";
let text2 = "Slight Injury";
let text3 = "Fatal injury";

let color1 = "#db8b23";
let color2 = "#06630c";
let color3 = "#db2f23";

let predictText = resultContainer.childNodes[1].innerHTML;

if(predictText.includes(text1)){
    resultContainer.style.background = color1;
}
else if(predictText.includes(text2)){
    resultContainer.style.background = color2;
}
else{
    resultContainer.style.background = color3;
}