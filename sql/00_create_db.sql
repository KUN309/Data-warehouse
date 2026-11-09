-- Script tạo CSDL credit_risk_dw cho môn CO4031
IF DB_ID('credit_risk_dw') IS NULL 
BEGIN
    CREATE DATABASE credit_risk_dw;
    PRINT 'Đã tạo CSDL credit_risk_dw thành công.';
END
ELSE
BEGIN
    PRINT 'CSDL credit_risk_dw đã tồn tại.';
END
GO

