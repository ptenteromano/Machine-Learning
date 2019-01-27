// a matrix multiplication algorithm
// for further solidifying this linear algebra concept

// only works on 2x2 for now...

// 2, 2x2 matrices
const m1 = [[1, 2], [3, 4]];
const m2 = [[5, 6], [7, 8]];
// const m1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
// const m2 = [[7], [8], [9]];

function matrixMult(a, b) {
  // num cols of first must === num rows of second
  if (a[0].length !== b.length) return "Cannot multiply these Matrices";

  let result = [];
  let inner = [];
  let temp,
    sum = 0,
    r = 0,
    rowCounter = 0;

  while (r < a.length) {
    for (let c = 0; c < a[r].length; c++) {
      temp = a[r][c] * b[c][rowCounter];
      console.log(a[r][c], b[c][rowCounter], temp);
      sum += temp;
    }
    rowCounter++;
    console.log(sum);
    inner.push(sum);
    sum = 0;
    if (rowCounter >= b.length) {
      r++;
      rowCounter = 0;
      result.push(inner);
      inner = [];
    }
  }
  return result;
}

let z = matrixMult(m1, m2);

console.log(z);
