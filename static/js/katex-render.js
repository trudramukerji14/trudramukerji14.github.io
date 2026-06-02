renderMathInElement(document.body, {
  delimiters: [
    {left: '$$', right: '$$', display: true},
    {left: '\\[', right: '\\]', display: true},
    {left: '\\begin{equation}', right: '\\end{equation}', display: true},
    {left: '\\begin{align}', right: '\\end{align}', display: true},
    {left: '$', right: '$', display: false},
    {left: '\\(', right: '\\)', display: false}
  ],
  trust: true
});
