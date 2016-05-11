set term pngcairo size 1200,800

set xrange [0:]

do for [s in "2 4 8 16 32 64 128 256 512 1024"] {
   set output s . ".png"
   plot s . ".bstats" using 1:2 with impulses
}
