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
(assert (let ((?v_7 (* 1 x4)) (?v_0 (* 0 x5)) (?v_26 (* 1 x1)) (?v_19 (* 1 x5)) (?v_1 (* 0 x8)) (?v_6 (* 1 x9)) (?v_3 (* 0 x6)) (?v_8 (* 0 x1)) (?v_4 (* 0 x9)) (?v_9 (* 0 x3)) (?v_11 (* 1 x8)) (?v_17 (* 1 x6)) (?v_21 (* 0 x4)) (?v_15 (* 1 x2)) (?v_2 (* 0 x7)) (?v_5 (* 0 x0)) (?v_28 (* 1 x7)) (?v_25 (* 1 x0)) (?v_18 (* 0 x2)) (?v_27 (* 1 x3)) (?v_10 (* (- 1) x1)) (?v_22 (* (- 1) x0)) (?v_16 (* (- 1) x9)) (?v_29 (* (- 1) x5)) (?v_12 (* (- 1) x3)) (?v_14 (* (- 1) x7)) (?v_20 (* (- 1) x8)) (?v_13 (* (- 1) x6)) (?v_24 (* (- 1) x2)) (?v_23 (* (- 1) x4))) (and (<= (+ ?v_7 ?v_5 ?v_3 ?v_2 ?v_10 ?v_0 ?v_26 ?v_22 ?v_16 ?v_0) 1) (<= (+ ?v_19 ?v_1 ?v_1 ?v_29 ?v_18 ?v_1 ?v_12 ?v_6 ?v_2 ?v_14) 1) (<= (+ ?v_3 ?v_8 ?v_4 ?v_4 ?v_9 ?v_11 ?v_5 ?v_6 ?v_0 ?v_7) 0) (<= (+ ?v_8 ?v_17 ?v_9 ?v_5 ?v_20 ?v_9 ?v_8 ?v_10 ?v_8 ?v_13) 1) (<= (+ ?v_21 ?v_5 ?v_15 ?v_11 ?v_12 ?v_13 ?v_2 ?v_14 ?v_15 ?v_16) 1) (<= (+ ?v_17 ?v_14 ?v_1 ?v_1 ?v_16 ?v_15 ?v_0 ?v_9 ?v_18 ?v_14) 0) (<= (+ ?v_19 ?v_18 ?v_3 ?v_24 ?v_1 ?v_4 ?v_20 ?v_5 ?v_0 ?v_5) (- 1)) (<= (+ ?v_0 ?v_1 ?v_11 ?v_10 ?v_28 ?v_9 ?v_9 ?v_21 ?v_18 ?v_1) (- 1)) (<= (+ ?v_19 ?v_12 ?v_11 ?v_3 ?v_8 ?v_22 ?v_23 ?v_25 ?v_7 ?v_1) (- 1)) (<= (+ ?v_5 ?v_1 ?v_5 ?v_3 ?v_23 ?v_4 ?v_21 ?v_24 ?v_16 ?v_10) (- 1)) (<= (+ ?v_17 ?v_3 ?v_9 ?v_8 ?v_3 ?v_5 ?v_5 ?v_21 ?v_25 ?v_21) 0) (<= (+ ?v_18 ?v_6 ?v_3 ?v_17 ?v_27 ?v_6 ?v_3 ?v_3 ?v_1 ?v_4) 0) (<= (+ ?v_26 ?v_1 ?v_2 ?v_6 ?v_2 ?v_9 ?v_4 ?v_13 ?v_19 ?v_18) 0) (<= (+ ?v_26 ?v_27 ?v_9 ?v_28 ?v_0 ?v_18 ?v_16 ?v_25 ?v_18 ?v_12) 0) (<= (+ ?v_26 ?v_5 ?v_2 ?v_14 ?v_10 ?v_23 ?v_5 ?v_18 ?v_20 ?v_9) (- 1)) (<= (+ ?v_9 ?v_2 ?v_21 ?v_22 ?v_3 ?v_18 ?v_1 ?v_8 ?v_3 ?v_20) 0) (<= (+ ?v_17 ?v_19 ?v_7 ?v_29 ?v_2 ?v_21 ?v_3 ?v_16 ?v_12 ?v_6) (- 1)) (<= (+ ?v_19 ?v_8 ?v_14 ?v_9 ?v_14 ?v_8 ?v_2 ?v_17 ?v_2 ?v_3) 0) (<= (+ ?v_3 ?v_13 ?v_7 ?v_18 ?v_5 ?v_6 ?v_19 ?v_11 ?v_6 ?v_27) 0) (<= (+ ?v_4 ?v_21 ?v_4 ?v_5 ?v_21 ?v_10 ?v_18 ?v_15 ?v_18 ?v_25) 0) (<= (+ ?v_11 ?v_0 ?v_7 ?v_3 ?v_1 ?v_26 ?v_10 ?v_17 ?v_19 ?v_29) (- 1)) (<= (+ ?v_28 ?v_3 ?v_18 ?v_3 ?v_21 ?v_29 ?v_3 ?v_18 ?v_18 ?v_13) (- 1)) (<= (+ ?v_5 ?v_14 ?v_29 ?v_26 ?v_27 ?v_8 ?v_1 ?v_3 ?v_24 ?v_15) 0) (<= (+ ?v_4 ?v_13 ?v_3 ?v_9 ?v_11 ?v_5 ?v_3 ?v_24 ?v_27 ?v_11) 0) (<= (+ ?v_9 ?v_21 ?v_8 ?v_26 ?v_4 ?v_0 ?v_28 ?v_24 ?v_9 ?v_5) 1))))
(check-sat)
(exit)
