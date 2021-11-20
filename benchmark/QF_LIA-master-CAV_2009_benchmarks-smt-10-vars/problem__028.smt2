(set-info :smt-lib-version 2.6)
(set-logic QF_LIA)
(set-info :source |
Alberto Griggio

|)
(set-info :category "random")
(set-info :status unsat)
(declare-fun x0 () Int)
(declare-fun x1 () Int)
(declare-fun x2 () Int)
(declare-fun x3 () Int)
(declare-fun x4 () Int)
(declare-fun x5 () Int)
(declare-fun x6 () Int)
(declare-fun x7 () Int)
(declare-fun x8 () Int)
(declare-fun x9 () Int)
(assert (let ((?v_3 (* 0 x5)) (?v_0 (* 1 x0)) (?v_1 (* 0 x1)) (?v_2 (* 0 x2)) (?v_13 (* 1 x4)) (?v_25 (* 1 x9)) (?v_5 (* 0 x4)) (?v_6 (* 0 x9)) (?v_12 (* 1 x8)) (?v_10 (* 0 x7)) (?v_11 (* 0 x6)) (?v_26 (* 1 x5)) (?v_21 (* 1 x6)) (?v_15 (* 1 x1)) (?v_18 (* 1 x2)) (?v_20 (* 1 x3)) (?v_16 (* 0 x3)) (?v_9 (* 0 x0)) (?v_28 (* 1 x7)) (?v_7 (* 0 x8)) (?v_29 (* (- 1) x9)) (?v_4 (* (- 1) x2)) (?v_8 (* (- 1) x8)) (?v_24 (* (- 1) x3)) (?v_14 (* (- 1) x6)) (?v_17 (* (- 1) x1)) (?v_19 (* (- 1) x7)) (?v_27 (* (- 1) x5)) (?v_22 (* (- 1) x0)) (?v_23 (* (- 1) x4))) (and (<= (+ ?v_3 ?v_0 ?v_0 ?v_1 ?v_29 ?v_2 ?v_1 ?v_13 ?v_6 ?v_4) 1) (<= (+ ?v_2 ?v_3 ?v_25 ?v_5 ?v_4 ?v_8 ?v_5 ?v_7 ?v_9 ?v_6) (- 1)) (<= (+ ?v_12 ?v_10 ?v_24 ?v_1 ?v_7 ?v_11 ?v_2 ?v_1 ?v_8 ?v_2) 0) (<= (+ ?v_1 ?v_9 ?v_14 ?v_17 ?v_19 ?v_0 ?v_0 ?v_16 ?v_5 ?v_26) 0) (<= (+ ?v_6 ?v_10 ?v_5 ?v_27 ?v_21 ?v_8 ?v_15 ?v_10 ?v_8 ?v_11) (- 1)) (<= (+ ?v_12 ?v_7 ?v_7 ?v_18 ?v_11 ?v_2 ?v_11 ?v_13 ?v_14 ?v_2) 0) (<= (+ ?v_2 ?v_15 ?v_14 ?v_20 ?v_0 ?v_8 ?v_9 ?v_10 ?v_6 ?v_11) 0) (<= (+ ?v_16 ?v_17 ?v_18 ?v_12 ?v_22 ?v_19 ?v_7 ?v_8 ?v_6 ?v_10) (- 1)) (<= (+ ?v_20 ?v_5 ?v_2 ?v_23 ?v_13 ?v_12 ?v_21 ?v_22 ?v_23 ?v_5) (- 1)) (<= (+ ?v_1 ?v_10 ?v_18 ?v_24 ?v_2 ?v_11 ?v_21 ?v_0 ?v_7 ?v_2) 0) (<= (+ ?v_12 ?v_7 ?v_25 ?v_5 ?v_26 ?v_12 ?v_2 ?v_5 ?v_9 ?v_4) 0) (<= (+ ?v_11 ?v_6 ?v_14 ?v_17 ?v_0 ?v_13 ?v_9 ?v_5 ?v_6 ?v_5) 1) (<= (+ ?v_28 ?v_6 ?v_16 ?v_22 ?v_15 ?v_23 ?v_22 ?v_16 ?v_18 ?v_27) 0) (<= (+ ?v_20 ?v_24 ?v_7 ?v_7 ?v_28 ?v_8 ?v_3 ?v_29 ?v_7 ?v_0) 0) (<= (+ ?v_13 ?v_11 ?v_6 ?v_6 ?v_25 ?v_24 ?v_17 ?v_2 ?v_24 ?v_21) 0) (<= (+ ?v_7 ?v_2 ?v_17 ?v_13 ?v_2 ?v_7 ?v_21 ?v_18 ?v_1 ?v_23) (- 1)) (<= (+ ?v_10 ?v_5 ?v_5 ?v_7 ?v_16 ?v_17 ?v_25 ?v_29 ?v_27 ?v_13) 1) (<= (+ ?v_10 ?v_10 ?v_26 ?v_25 ?v_22 ?v_2 ?v_8 ?v_14 ?v_1 ?v_11) (- 1)) (<= (+ ?v_20 ?v_9 ?v_21 ?v_10 ?v_9 ?v_9 ?v_6 ?v_16 ?v_12 ?v_18) (- 1)) (<= (+ ?v_2 ?v_16 ?v_28 ?v_10 ?v_3 ?v_10 ?v_18 ?v_11 ?v_25 ?v_15) 0) (<= (+ ?v_0 ?v_3 ?v_11 ?v_27 ?v_2 ?v_5 ?v_5 ?v_6 ?v_29 ?v_6) 1) (<= (+ ?v_20 ?v_6 ?v_3 ?v_12 ?v_16 ?v_9 ?v_9 ?v_19 ?v_16 ?v_14) 1) (<= (+ ?v_6 ?v_14 ?v_5 ?v_17 ?v_6 ?v_7 ?v_11 ?v_4 ?v_12 ?v_23) 1) (<= (+ ?v_0 ?v_4 ?v_21 ?v_24 ?v_9 ?v_7 ?v_3 ?v_13 ?v_24 ?v_29) (- 1)) (<= (+ ?v_5 ?v_9 ?v_0 ?v_11 ?v_2 ?v_5 ?v_7 ?v_2 ?v_14 ?v_9) 0) (<= (+ ?v_2 ?v_4 ?v_28 ?v_1 ?v_3 ?v_11 ?v_13 ?v_13 ?v_0 ?v_1) 0) (<= (+ ?v_7 ?v_26 ?v_2 ?v_22 ?v_3 ?v_3 ?v_4 ?v_29 ?v_28 ?v_26) 0) (<= (+ ?v_18 ?v_3 ?v_14 ?v_26 ?v_7 ?v_22 ?v_28 ?v_15 ?v_6 ?v_15) 1) (<= (+ ?v_15 ?v_7 ?v_9 ?v_16 ?v_5 ?v_10 ?v_21 ?v_6 ?v_24 ?v_21) 0) (<= (+ ?v_13 ?v_23 ?v_2 ?v_17 ?v_23 ?v_14 ?v_22 ?v_3 ?v_6 ?v_10) 0))))
(check-sat)
(exit)
