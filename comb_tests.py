
def iterative_combs(numberTests):
    couples_combs = []
    ## numberTests = 10
    
    not_include = False
    
    for x in range(0, numberTests):
        for y in range(0, numberTests):
            if x != y:
                
                couple_number_tests = (x,y)           
                
                if len(couples_combs) > 0:
                    for x in range(0, numberTests):
                        for y in range(0, numberTests):
                            couple_x = (x,y)
                            if couple_number_tests[0] == couple_x[1] and couple_number_tests[1] == couple_x[0]:
                                not_include = True
                
                if not not_include:
                    couples_combs.append(couple_number_tests)
                    proc_algorithm(x, y)
                else:
                    not_include = True
			 