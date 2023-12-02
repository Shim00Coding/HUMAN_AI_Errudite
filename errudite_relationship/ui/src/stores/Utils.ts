
import * as d3 from 'd3';
import { ClassDatum, Token } from './Interfaces';
import * as jsdiff from 'diff';

interface AnswerColor {
    dark: string;
    light: string;
}

class UtilsClass {
    public F1Colors: string[]; // the F1 colors
    public selectedModelColor: string;
    public F1ColorScale: d3.ScaleQuantile<string>; // the F1 color scale

    public answerColor: {
        groundtruth: AnswerColor;
        correct: AnswerColor;
        incorrect: AnswerColor;
    };
    public correctThreshold: number;

    constructor() {
        // from blue to orange, drark to light
        // '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'
        // '#e6550d', '#fd8d3c', '#fdae6b', '#fdd0a2'
        this.correctThreshold = 1; // this is the threshold for the prediction being correct or incorrect
        this.F1Colors = [
            '#e6550d', '#fdae6b', '#fd8d3c',
            '#c6dbef', '#9ecae1', '#6baed6'
        ];
        this.F1ColorScale = d3.scaleQuantile<string>()
            .domain([0, 1.01]).range(this.F1Colors);
        this.selectedModelColor = 'steelblue';
            //.range(['#911a24', '#e6959c', '#db646f']);

        this.answerColor = {
            groundtruth: {dark: 'black', light: '#d9d9d9'},
            correct: {dark: '#2db7f5', light: '#9ecae1'}, // blue
            incorrect: {dark: '#e6550d', light: '#fdae6b'}, // orange
        }
    }


    public percent(n: number): string {
        const p = d3.format(".1%");
        return p(n);
    }

    public getAttr(obj: any, name: string, defaultReturn: any=null) {
        if ((name in obj)) { return obj[name]; }
        else { return defaultReturn; }
    }

    /**
     * Generate class for everything.
     * @param viewType      <String>    tree|network|item
     * @param elementType   <String>    node|group|text|link|keyword|cluster
     * @param identifier    <String>    info to identify one single item
     */
    public genClass (viewType: string, elementType: string,
                     identifier: string|number|(string|number)[]): ClassDatum {
        let keystring: string = '';
        switch (typeof(identifier)) {
            case 'string': keystring = <string>identifier; break;
            case 'number': keystring = (<number>identifier).toString(); break;
            default:
                const identifierArr = (<(number|string)[]>identifier).map((i) => {
                    return typeof(i) === 'string' ? i : i.toString();
                });
                keystring = identifierArr.join('-');
        }
        const element = `${viewType}-${elementType}`;
        const key = `${element}-${keystring}`;
        return {
            view: viewType,
            element: element,
            key: key,
            total: `${viewType} ${elementType} ${element} ${key}`
        };
    };

    public objEqual( x: object, y: object): boolean {
        if ( x === y ) { return true; };
        // if both x and y are null or undefined and exactly the same
        if ( !( x instanceof Object ) || !( y instanceof Object ) )  { return false };
        // if they are not strictly equal, they both need to be Objects
        if ( x.constructor !== y.constructor ) { return false; }
        // they must have the exact same prototype chain, the closest we can do is
        // test there constructor.
        // tslint:disable-next-line:no-for-in
        for (let i = 0; i < Object.keys(x).length; i++) {
            const p = Object.keys(x)[i];
            if ( ! x.hasOwnProperty( p ) ) { continue; }
            // other properties were tested using x.constructor === y.constructor
            if ( ! y.hasOwnProperty( p ) ) { return false; }
            // allows to compare x[ p ] and y[ p ] when set to undefined
            if ( x[ p ] === y[ p ] ) { continue; }
            // if they have the same strict value or identity then they are equal
            if ( typeof( x[ p ] ) !== 'object' ) { return false; }
            // Numbers, Strings, Functions, Booleans must be strictly equal
            if ( ! this.objEqual( x[ p ],  y[ p ] ) ) { return false; }
            // Objects and Arrays must be tested recursively
        }
        Object.keys(y).forEach(p => {
            if ( y.hasOwnProperty( p ) && ! x.hasOwnProperty( p ) ) { return false; }
            // allows x[ p ] to be set to undefined
        });
        return true;
    }

    /**
     * Filter and find unique instances in an array
     * Only used to be put in the filter.
     */
    public uniques(value: any, index: any, self: any[]): boolean {
        return self.indexOf(value) === index;
    }

    // group array into hash list by a key
    public groupBy<T> (xs: T[], key: string): {[key: string]: T[]} {
        return xs.reduce((rv, x) => {
            (rv[x[key]] = rv[x[key]] || []).push(x);
            return rv;
        // tslint:disable-next-line:align
        }, { });
    };
    /**
     * Return all permutations of 'length' elements from array:
     * This is a generator function
     * @param array <any[]> just a given array
     * @param length <number> the number the elements in the returned permutations.
     */
    public* permute(array: any[], length: number): any {
        if (length < 1) {
            yield [];
        } else {
            for (const element of array) {
                for (const combination of this.permute(array, length - 1)) {
                    yield combination.concat(element);
                }
            }
        }
    }

    /**
     * Generate all bit combinations for a given number [[true, true, true], ...]
     * Mostly used to generate which model is included when building error intersections.
     * @param count <int> the number of bits needed.
     */
    public genBits(count: number): boolean[][] {
        // tslint:disable-next-line:no-bitwise
        const bool2d: boolean[][] = [];
        // tslint:disable-next-line:no-bitwise
        for (let i = 0; i < (1 << count); i++) {
            const boolArr: boolean[] = [];
            //Increasing or decreasing depending on which direction
            //you want your array to represent the binary number
            for (let j = count - 1; j >= 0; j--) {
                // tslint:disable-next-line:no-bitwise
                boolArr.push(Boolean(i & (1 << j)));
            }
            bool2d.push(boolArr);
        }
        return bool2d;
    }
    public genStrId (key: string|number): string {
        if (typeof key === 'string') {
            return key.replace(/[^a-zA-Z1-9]/g, '-');
        } else {
            return 'number';
        }
    }

    public transpose (a: any[]): any[] {
        return a && a.length && a[0].map && a[0].map((_: any, c: any) => a.map(r => r[c]) || []);
    }
    public argMax(array: number[]) {
        return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
    }

    public clone(obj) {
        let copy;

        // Handle the 3 simple types, and null or undefined
        if (null == obj || "object" != typeof obj) return obj;

        // Handle Date
        if (obj instanceof Date) {
            copy = new Date();
            copy.setTime(obj.getTime());
            return copy;
        }

        // Handle Array
        if (obj instanceof Array) {
            copy = [];
            for (var i = 0, len = obj.length; i < len; i++) {
                copy[i] = this.clone(obj[i]);
            }
            return copy;
        }

        // Handle Object
        if (obj instanceof Object) {
            copy = {};
            for (let attr in obj) {
                if (obj.hasOwnProperty(attr)) copy[attr] = this.clone(obj[attr]);
            }
            return copy;
        }

        throw new Error("Unable to copy obj! Its type isn't supported.");
    }

    public genKeywordId (key: string): string {
        return key.replace(/[^a-zA-Z0-9]/g, '-');
    }

    public intersection(multi_array: any[][]): any[] {
        var result = [];
      var lists = multi_array;
      
      for(var i = 0; i < lists.length; i++) {
          var currentList = lists[i];
          for(var y = 0; y < currentList.length; y++) {
            var currentValue = currentList[y];
          if(result.indexOf(currentValue) === -1) {
            var existsInAll = true;
            for(var x = 0; x < lists.length; x++) {
              if(lists[x].indexOf(currentValue) === -1) {
                existsInAll = false;
                break;
              }
            }
            if(existsInAll) {
              result.push(currentValue);
            }
          }
        }
      }
      return result;
    }

    /**
     * Compute the rewrites.
     * @param {Token[]} oldToken the token series before rewritting
     * @param {Token[]} newToken the token series after rewritting
     * @return get the rewritten serives.
     */
    public computeRewrite(oldToken: Token[], newToken: Token[]): Token[] {
        const oldArr = oldToken.map(t => t.text);
        const newArr = newToken.map(t => t.text);
        const rawDiff = jsdiff.diffArrays(oldArr, newArr);
        // construct the tokenDatum
        let newIdx = 0;
        let oldIdx = 0;
        const tokens: Token[] = [];
        rawDiff.forEach((diff: {added: boolean, count: number, removed: boolean, value: string[]}) => {
            diff.value.forEach((t: string) => {
                const idx = diff.removed ? oldIdx : newIdx;
                const curT = diff.removed ? oldToken[idx] : newToken[idx];
                const etype = diff.removed ? 'remove' : diff.added ? 'add' : 'keep';
                const curToken: Token = {
                    text: curT.text, idx: idx,
                    sid: curT.sid,
                    ner: curT.ner, pos: curT.pos, tag: curT.tag,
                    lemma: curT.lemma, whitespace: curT.whitespace,
                    etype: etype,
                    matches: curT.matches
                };
                tokens.push(curToken);
                
                newIdx = diff.removed ? newIdx : newIdx + 1;
                oldIdx = diff.added ? oldIdx : oldIdx + 1;
            });
        });
        return tokens;
    }

    public computeRewriteStr(atext: string, btext: string): Token[] {
        const oldArr = atext.split(' ');
        const newArr = btext.split(' ');
        const rawDiff = jsdiff.diffArrays(oldArr, newArr);
        // construct the tokenDatum
        let newIdx = 0;
        let oldIdx = 0;
        const tokens = [];
        rawDiff.forEach((diff: {added: boolean, count: number, removed: boolean, value: string[]}) => {
            diff.value.forEach((t: string) => {
                const idx = diff.removed ? oldIdx : newIdx;
                const curT = diff.removed ? oldArr[idx] : newArr[idx];
                const etype: 'remove'|'add'|'keep' = diff.removed ? 'remove' : diff.added ? 'add' : 'keep';
                const curToken = {
                    text: curT, etype: etype, idx: idx
                };
                tokens.push(curToken);
                newIdx = diff.removed ? newIdx : newIdx + 1;
                oldIdx = diff.added ? oldIdx : oldIdx + 1;
            });
        });
        return tokens;
    }


    /**
     * Make the instances into raw text
     * @return plain text of the instance.
     */
    public textize(doc: Token[]): string {
        let text: string = '';
        doc.forEach((d: Token) => { text += d.text + d.whitespace; });
        return text;
    }
    public pad(num, size): string {
        var s = num+"";
        while (s.length < size) s = "0" + s;
        return s;
    }
}

export const utils = new UtilsClass();



