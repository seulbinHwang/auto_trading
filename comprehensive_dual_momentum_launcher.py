import sys
"""
- QApplication
    - PyQt5 를 이용하여 API 를 제어하는 메인 루프
    - OCX 방식인 API를 사용할 수 있게 됨
"""
if len(sys.argv) == 2:
    additional_investment_amount = int(sys.argv[1])
else:
    additional_investment_amount = 0

use_kiwoom = False
if use_kiwoom:
    app = QApplication(sys.argv)
    from strategy.comprehensive_dual_momentum_strategy import ComprehensiveDualMomentumSrategyWKiwoom

    comprehensive_dual_momentum_strategy = ComprehensiveDualMomentumSrategyWKiwoom()
    comprehensive_dual_momentum_strategy.start()
    app.exec_()
else:
    from strategy.comprehensive_dual_momentum_strategy import ComprehensiveDualMomentumSrategy
    comprehensive_dual_momentum_strategy = ComprehensiveDualMomentumSrategy(additional_investment_amount, add_chore_universe=False)
    comprehensive_dual_momentum_strategy.run_finance_data_reader()

