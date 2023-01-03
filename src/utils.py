import pandas as pd

def transform_ds_101_to_4(path_ds_101="../data/Codeflaws_features_Eq_101.parquet"):
    X = pd.read_parquet(path_ds_101)

    astp_enc = {'zext-ASTp': 2.3095505, 'getelementptr-ASTp': -0.03688684, 'fsub-ASTp': -0.597503,
                'add-ASTp': -0.10737835, 'and-ASTp': 0.8570806, 'fcmp-ASTp': -0.59506524, 'fmul-ASTp': -0.583374,
                'mul-ASTp': 0.09424518, 'br-ASTp': 0.5566651, 'icmp-ASTp': -0.60860205, 'call-ASTp': -0.59501714,
                'select-ASTp': 2.2194602, 'sdiv-ASTp': -0.00239579, 'sub-ASTp': 0.09151424, 'srem-ASTp': -0.214511,
                'sext-ASTp': 0.6889393, 'store-ASTp': -0.5855826, 'trunc-ASTp': 1.182849, 'fadd-ASTp': -0.05108408,
                'ret-ASTp': -0.02686079}
    K_astp = list(astp_enc)

    stmt_enc = {'true-BBType': -0.80127406, 'false-BBType': -0.73828125, 'rhs-BBType': -0.66262877,
                'lor-BBType': -0.08475034, 'for-BBType': 0.49366575, 'then-BBType': -0.70792705,
                'if-BBType': -0.04038725, 'while-BBType': -0.7583362, 'land-BBType': -0.6561358,
                'else-BBType': -0.34240085, 'lhs.false-BBType': -0.69320023, 'lhs.true-BBType': 0.03849571,
                'body-BBType': -0.5256219, 'inc-BBType': -0.21445361, 'cond-BBType': 0.0677571,
                'end-BBType': 0.03480222, 'entry-BBType': 0.0448336}
    K_stmt = list(stmt_enc)

    others_enc = {'@-Matcher-astparent': -0.007498912, '10-Operand-DataTypeContext': -0.022472369,
                  '@-Matcher-inctrldep': 1.6135985, '@-Matcher-outdatadep': 4.5306482, '@-Matcher-indatadep': 11.683311,
                  '10-Return-DataTypeContext': -0.020886639}
    K_others = list(others_enc)

    matchers = ['BITSHL$@1$@2$-Matcher', 'PLEFTINC$P1$-Matcher', 'BITXOR$@1$@2$-Matcher', 'DIV$@1$@2$-Matcher',
                'BITSHR$@1$@2$-Matcher', 'ADD$@1$@2$-Matcher', 'LT$@1$@2$-Matcher', 'C-Matcher', 'AND$@1$@2$-Matcher',
                'GE$@1$@2$-Matcher', 'BITOR$@1$@2$-Matcher', 'PADD$A1$@2$-Matcher', 'GT$@1$@2$-Matcher',
                'LE$@1$@2$-Matcher', 'ASSIGN$V1$@2$-Matcher', 'SUB$@1$@2$-Matcher', 'BITAND$@1$@2$-Matcher',
                'EQ$@1$@2$-Matcher', 'NEQ$@1$@2$-Matcher', 'STMT-Matcher', 'MOD$@1$@2$-Matcher', 'A-Matcher',
                'MUL$@1$@2$-Matcher', 'PADD_DEREF$A1$@2$-Matcher', 'PADD_DEREF$P1$@2$-Matcher']

    replacers = ['EQ$@1$@2$-Replacer', 'GE$@1$@2$-Replacer', 'GT$@2$@1$-Replacer', 'OPERAND$@2$-Replacer',
                 'GE$@2$@1$-Replacer', 'MOD$@1$@2$-Replacer', 'BITOR$@1$@2$-Replacer', 'BITSHL$@2$@1$-Replacer',
                 'BITSHR$@1$@2$-Replacer', 'CONSTVAL$0$-Replacer', 'ABS$@1$-Replacer', 'DIV$@2$@1$-Replacer',
                 'LE$@2$@1$-Replacer', 'BITXOR$@1$@2$-Replacer', 'MOD$@2$@1$-Replacer', 'BITNOT$@1$-Replacer',
                 'NEQ$@1$@2$-Replacer', 'SUB$@2$@1$-Replacer', 'NEG$@2$-Replacer', 'ASSIGN$V2$@1$-Replacer',
                 'SUB$@1$@2$-Replacer', 'ASSIGN$V1$@2$-Replacer', 'SHUFFLEARGS$2$-Replacer', 'ABS$@$-Replacer',
                 'LT$@2$@1$-Replacer', 'LE$@1$@2$-Replacer', 'BITNOT$@2$-Replacer', 'ABS$@2$-Replacer',
                 'DELSTMT-Replacer', 'TRAPSTMT-Replacer', 'DIV$@1$@2$-Replacer', 'GT$@1$@2$-Replacer',
                 'PADD$A1$@2$-Replacer']

    X["Matcher"] = (X[matchers].apply(lambda x: '|'.join([matchers[i] for i in range(len(x)) if x[i] == 1]), axis=1)).apply(lambda y: 'Others-Matcher' if y == '' else y)
    X["Replacer"] = (X[replacers].apply(lambda x: '|'.join([replacers[i] for i in range(len(x)) if x[i] == 1]), axis=1)).apply(lambda y: 'Others-Replacer' if y == '' else y)
    X["Others"] = X[K_others].apply(lambda x: sum([x[idx] * others_enc[K_others[idx]] for idx in range(len(K_others))]),axis=1)
    X["Stmt"] = X[K_stmt].apply(lambda x: sum([x[idx] * stmt_enc[K_stmt[idx]] for idx in range(len(K_stmt))]), axis=1)
    X["Astp"] = X[K_astp].apply(lambda x: sum([x[idx] * astp_enc[K_astp[idx]] for idx in range(len(K_astp))]), axis=1)
    X["MuOp"] = X[["Matcher", "Replacer"]].apply(lambda x: x[0] + "|" + x[1], axis=1)

    X2 = pd.DataFrame()
    X2["MuOp"] = X["MuOp"]
    X2["Stmt"] = X["Stmt"]
    X2["Astp"] = X["Astp"]
    X2["Others"] = X["Others"]
    X2["Eq"] = X["Eq"]

    return X2