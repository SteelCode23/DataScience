


Select customer.CustomerID, sales.CustomerPurchases, customer.Name

FROM customer join

       (

Select CustomerID, COUNT(Orders) CustomerPurchases

       From sales

       GROUP BY CustomerID

       ) sales join customer on sales.CustomerID = customer.CustomerID



;with sales as (

Select CustomerID, COUNT(Orders) CustomerPurchases

       From sales

       GROUP BY CustomerID

       )

Select customer.CustomerID, sales.CustomerPurchases, customer.Name

FROM customer join sales join customer on sales.CustomerID = customer.CustomerID



DROP TABLE IF EXISTS #Sales

Select CustomerID, COUNT(Orders) CustomerPurchases

INTO #SALES

From sales

GROUP BY CustomerID

CREATE NONCLUSTERED INDEX i1 ON #SALES(CustomerID)

Select customer.CustomerID, sales.CustomerPurchases, customer.Name

FROM customer join #SALES sales join customer on sales.CustomerID = customer.CustomerID


DECLARE @Month VARCHAR(MAX)

SET @Month = 'June'

DECLARE @SQL VARCHAR(MAX)

SET @SQL = '''

SELECT ''' + @Month + '''

INTO ##Date

FROM DateTable

'''

EXEC(@SQL)
